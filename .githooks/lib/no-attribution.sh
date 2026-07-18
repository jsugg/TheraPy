# TheraPy hooks — context-aware no-attribution predicate. POSIX sh, zero-dep.
# Blocks a TOOL crediting itself as author; ALLOWS plain vendor mention.
#   passes:  "anthropic adapter ...", "fixtures for anthropic/openrouter/ollama"
#   blocks:  robot-emoji footers, "Co-authored-by:" trailers, "<verb> by/with <tool>"
#
# Usage:  hits=$(printf '%s\n' "$text" | attr_scan);  [ -n "$hits" ] && reject
# Prints one "reason  offending-line" per signal; empty output = clean.
# To disable the guard entirely: delete this file. The hooks source it only when
# present, so removing it turns the check off without breaking commits/pushes.

# Tool/agent identities — matched ONLY inside an attribution context (below),
# never on their own, so a bare vendor mention never trips.
attr_AGENTS='(ai|llm|gpt|bot|copilot|assistant|agent|model|claude|anthropic|codex|openai|chatgpt|gemini|cursor|codeium|devin)'
# Product names that are themselves the attribution when they stand alone.
attr_BRANDS='(claude|anthropic|codex|openai|chatgpt|copilot|gemini|cursor|codeium|devin|gpt)'
# Verbs asserting the tool produced the work.
attr_VERBS='(generated|written|created|authored|produced|composed|drafted|assisted|refactored|implemented|co-?authored|co-?written)'

attr_scan() {
  _t=$(cat)
  printf '%s\n' "$_t" | grep -nE '🤖' | sed 's/^/robot-emoji  /' || true
  printf '%s\n' "$_t" | grep -niE '^[[:space:]]*(co-?authored|co-?written)[- ](by|with)[[:space:]]*:' | sed 's/^/co-author    /' || true
  printf '%s\n' "$_t" | grep -iE "${attr_VERBS}[- ]?(by|with|using|via)" | grep -iE "(^|[^[:alnum:]])${attr_AGENTS}([^[:alnum:]]|$)" | sed 's/^/authored-by  /' || true
  printf '%s\n' "$_t" | grep -iE '(powered|built|crafted|made)[- ](by|with)' | grep -iE "(^|[^[:alnum:]])${attr_AGENTS}([^[:alnum:]]|$)" | sed 's/^/promo        /' || true
  printf '%s\n' "$_t" | grep -niE 'claude\.(ai|com)|anthropic\.com|chatgpt\.com|chat\.openai\.com|/(github-)?copilot|cursor\.(com|sh)|codeium\.com|noreply@(anthropic|openai)\.com' | sed 's/^/tool-url     /' || true
  printf '%s\n' "$_t" | grep -niE "^[[:space:]]*(--|—|–|~)[[:space:]]*${attr_BRANDS}[[:space:]]*$" | sed 's/^/signature    /' || true
}
