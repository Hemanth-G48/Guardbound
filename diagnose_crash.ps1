<#
  Guardbound - crash diagnosis helper (Phase 14 HYPERVISOR_ERROR investigation)

  READ-ONLY BY DEFAULT. Nothing on the system is changed unless you pass -Apply.

  Run from an ELEVATED PowerShell prompt (the minidumps and WER reports are
  ACL-protected, so a normal prompt gets "Access is denied").

    powershell -ExecutionPolicy Bypass -File .\diagnose_crash.ps1
        -> collects evidence only. Safe. Do this first.

    powershell -ExecutionPolicy Bypass -File .\diagnose_crash.ps1 -Apply
        -> ALSO enables Automatic kernel dumps + a dedicated dump file on D:\,
           so the NEXT crash produces an analyzable dump.

    powershell -ExecutionPolicy Bypass -File .\diagnose_crash.ps1 -Apply -NoAutoReboot
        -> ...and stops the auto-restart so you actually see the BSOD stop code.

  Background: the machine bugchecks with 0x00020001 HYPERVISOR_ERROR and is
  configured with CrashDumpEnabled=3 (small minidump only), so no kernel dump
  or MEMORY.DMP is ever produced.
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$NoAutoReboot,
    [string]$OutDir = "$env:USERPROFILE\Desktop\Guardbound_crash_dumps",
    [string]$DedicatedDumpPath = "D:\dedicateddump.sys",
    [int]$DedicatedDumpSizeMB = 40960
)

$ErrorActionPreference = 'Continue'
$CK = 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl'

function Section($t) { Write-Host ''; Write-Host ('=' * 78); Write-Host $t; Write-Host ('=' * 78) }

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Section '0. CONTEXT'
Write-Host ("Elevated          : {0}" -f $isAdmin)
Write-Host ("Output directory  : {0}" -f $OutDir)
Write-Host ("Apply mode        : {0}" -f $Apply)
if (-not $isAdmin) {
    Write-Host 'WARNING: not elevated. Minidump/WER copy and -Apply will fail.' -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
Section '1. CRASH DUMP CONFIGURATION (current)'
# ---------------------------------------------------------------------------
if (Test-Path $CK) {
    Get-ItemProperty $CK |
        Select-Object CrashDumpEnabled, DumpFile, MinidumpDir, DedicatedDumpFile,
                      DumpFileSize, AutoReboot, Overwrite, IgnorePagefileSize,
                      AlwaysKeepMemoryDump, LogEvent |
        Format-List
    Write-Host 'CrashDumpEnabled: 0=None 1=Complete 2=Kernel 3=Small(minidump) 7=Automatic'
}

# ---------------------------------------------------------------------------
Section '2. COLLECT CRASH EVIDENCE'
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host ("Created/exists: {0}" -f $OutDir)

$dumps = Get-ChildItem 'C:\Windows\Minidump' -Filter '*.dmp' -ErrorAction SilentlyContinue
Write-Host ("Minidumps found: {0}" -f $dumps.Count)
foreach ($d in $dumps) {
    try {
        Copy-Item $d.FullName -Destination $OutDir -Force -ErrorAction Stop
        Write-Host ("  copied {0} ({1:N0} bytes)" -f $d.Name, $d.Length)
    } catch {
        Write-Host ("  FAILED {0}: {1}" -f $d.Name, $_.Exception.Message) -ForegroundColor Red
    }
}

if (Test-Path 'C:\Windows\MEMORY.DMP') {
    Write-Host 'MEMORY.DMP present - copying'
    Copy-Item 'C:\Windows\MEMORY.DMP' -Destination $OutDir -Force
} else {
    Write-Host 'MEMORY.DMP: ABSENT (expected - CrashDumpEnabled=3 never writes one)' -ForegroundColor Yellow
}

# WER kernel reports (these carry the "possibly related driver" attribution)
$werRoot = 'C:\ProgramData\Microsoft\Windows\WER\ReportArchive'
$werKernel = Get-ChildItem $werRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'Kernel_*' }
Write-Host ("WER Kernel reports found: {0}" -f $werKernel.Count)
foreach ($w in $werKernel) {
    try {
        Copy-Item $w.FullName -Destination $OutDir -Recurse -Force -ErrorAction Stop
    } catch {
        Write-Host ("  FAILED {0}: {1}" -f $w.Name, $_.Exception.Message) -ForegroundColor Red
    }
}

# Bugcheck history -> text file for reference
$bugchecks = Get-WinEvent -FilterHashtable @{
    LogName   = 'System'
    ProviderName = 'Microsoft-Windows-WER-SystemErrorReporting'
} -ErrorAction SilentlyContinue
$attrib = Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = 1019 } -ErrorAction SilentlyContinue
$hist = @()
$hist += '=== BUGCHECK EVENTS (Id 1001) ==='
$hist += ($bugchecks | Sort-Object TimeCreated |
          ForEach-Object { '{0} | {1}' -f $_.TimeCreated, ($_.Message -replace "`r`n", ' ') })
$hist += ''
$hist += '=== DRIVER ATTRIBUTION (Id 1019) ==='
$hist += ($attrib | Sort-Object TimeCreated |
          ForEach-Object { '{0} | {1}' -f $_.TimeCreated, ($_.Message -replace "`r`n", ' ') })
$hist | Set-Content (Join-Path $OutDir 'bugcheck_history.txt') -Encoding UTF8
Write-Host 'Wrote bugcheck_history.txt'

# ---------------------------------------------------------------------------
Section '3. HYPERVISOR / SECURITY STACK STATE'
# ---------------------------------------------------------------------------
Write-Host '-- Hypervisors present --'
foreach ($n in @('uxen', 'hvservice', 'Vid', 'HvHost')) {
    $d = Get-CimInstance Win32_SystemDriver -Filter "Name='$n'" -ErrorAction SilentlyContinue
    if ($d) { Write-Host ("  {0,-10} State={1,-8} Started={2,-6} {3}" -f $d.Name, $d.State, $d.Started, $d.PathName) }
}
Write-Host '-- HP Sure Click / Wolf Security services --'
Get-Service -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(Br|uxen|hpsvc|HPApp|HPDiags|HPNetwork|HPSysInfo|hpqcaslwmiex)' } |
    Select-Object Status, StartType, Name, DisplayName |
    Format-Table -AutoSize | Out-String -Width 200 | Write-Host
Write-Host '-- Micro-VM processes --'
Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'Br-|uxen|BemSvc' } |
    Select-Object Name, Id, @{n = 'WS_MB'; e = { [math]::Round($_.WorkingSet64 / 1MB, 0) } }, StartTime |
    Format-Table -AutoSize | Write-Host

# ---------------------------------------------------------------------------
Section '4. FIRMWARE / DRIVER INVENTORY'
# ---------------------------------------------------------------------------
Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion, ReleaseDate | Format-List
Get-CimInstance Win32_VideoController |
    Select-Object Name, DriverVersion, DriverDate, Status | Format-Table -AutoSize

# ---------------------------------------------------------------------------
Section '5. DUMP TARGET VIABILITY'
# ---------------------------------------------------------------------------
Get-Volume -ErrorAction SilentlyContinue |
    Where-Object { $_.DriveLetter } |
    Select-Object DriveLetter, FileSystemLabel, HealthStatus,
                  @{n = 'FreeGB'; e = { [math]::Round($_.SizeRemaining / 1GB, 1) } } |
    Format-Table -AutoSize

# ---------------------------------------------------------------------------
if ($Apply) {
    Section '6. APPLYING DUMP-CAPTURE FIX'
    if (-not $isAdmin) {
        Write-Host 'SKIPPED - not elevated.' -ForegroundColor Red
    } else {
        $targetDrive = (Split-Path $DedicatedDumpPath -Qualifier)
        $vol = Get-Volume -DriveLetter $targetDrive.TrimEnd(':') -ErrorAction SilentlyContinue
        if (-not $vol) {
            Write-Host ("Target volume {0} not found - aborting." -f $targetDrive) -ForegroundColor Red
        } elseif ($vol.SizeRemaining -lt (($DedicatedDumpSizeMB + 5120) * 1MB)) {
            Write-Host 'Not enough free space on the dump target - aborting.' -ForegroundColor Red
        } else {
            Write-Host ("Enabling Automatic kernel dumps, dedicated file {0} ({1} MB)" -f `
                $DedicatedDumpPath, $DedicatedDumpSizeMB)
            Set-ItemProperty $CK -Name CrashDumpEnabled  -Value 7 -Type DWord
            Set-ItemProperty $CK -Name DedicatedDumpFile -Value $DedicatedDumpPath -Type String
            Set-ItemProperty $CK -Name DumpFileSize      -Value $DedicatedDumpSizeMB -Type DWord
            Set-ItemProperty $CK -Name Overwrite         -Value 1 -Type DWord

            if ($NoAutoReboot) {
                Write-Host 'Setting AutoReboot=0 (machine will sit at the BSOD screen)' -ForegroundColor Yellow
                Set-ItemProperty $CK -Name AutoReboot -Value 0 -Type DWord
            }
            Write-Host ''
            Write-Host 'New configuration:' -ForegroundColor Green
            Get-ItemProperty $CK |
                Select-Object CrashDumpEnabled, DumpFile, DedicatedDumpFile,
                              DumpFileSize, AutoReboot | Format-List
            Write-Host 'NOTE: takes effect immediately for the next bugcheck.' -ForegroundColor Green
            Write-Host 'NOTE: a 40 GB dump on D:\ may take a minute or two to write at crash time.' -ForegroundColor Yellow
        }
    }
} else {
    Section '6. NOT APPLIED'
    Write-Host 'Read-only run. Re-run with -Apply to enable Automatic kernel dumps.' -ForegroundColor Yellow
}

Section 'DONE'
Write-Host ("Evidence collected in: {0}" -f $OutDir)
Write-Host ''
Write-Host 'Next step - install a debugger and analyze the newest minidump:'
Write-Host '  winget install Microsoft.WinDbg'
Write-Host '  kd -z <newest .dmp> -c "!analyze -v; q"'
Write-Host '  (or open the .dmp in the WinDbg GUI and read the !analyze -v output)'
Write-Host ''
Write-Host 'Look specifically for: the FAULTING_MODULE, the 0x20001 subcode meaning,'
Write-Host 'and any stack frame naming uxen.sys / nvlddmkm.sys / igdkmdn64.sys.'
