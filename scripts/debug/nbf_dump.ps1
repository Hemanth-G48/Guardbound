$paths = @(
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/steering.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/crescendo/run.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/crescendo/prompts.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/opposite_day/run.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/opposite_day/prompts.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/acronym/run.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/acronym/prompts.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/actor_attack/run.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/actor_attack/prompts.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/generate.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/refusal.py',
 'nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/safety_score.py'
);
foreach($p in $paths){
 Write-Host '============================================================';
 Write-Host 'FILE:' $p;
 Write-Host '============================================================';
 Get-Content $p -Raw
}
