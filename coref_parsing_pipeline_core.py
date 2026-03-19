import spacy
import re
import gender_guesser.detector as gender

# --- INITIALIZATION ---
nlp = spacy.load("en_core_web_sm")
if "fastcoref" not in nlp.pipe_names:
    nlp.add_pipe("fastcoref")

# Normalize Unicode curly quotes → straight quotes so all quote
# detection logic only needs to handle one style.
def normalize_quotes(text):
    """
    Converts Unicode curly quotes to straight quotes so all downstream
    quote-detection logic only needs to handle one style.

    Example:
        Input:  \u201cI found it,\u201d she said.
        Output: "I found it," she said.
    """
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    return text


# =============================================================
# PHASE 0 — TEXT PREPROCESSING
# =============================================================
def preprocess_text(raw_text):
    """
    Cleans whitespace, normalizes quotes, splits into sentences, and prints them.

    Example:
        Input:  '  Emma  walked in.  She smiled. '
        Output: 'Emma walked in. She smiled.'
                S0: Emma walked in.
                S1: She smiled.
    """
    clean_text = normalize_quotes(re.sub(r'\s+', ' ', raw_text).strip())

    doc = nlp(clean_text)
    print("\n=== [PHASE 0] Pre-processed Original Sentences ===")
    for i, s in enumerate(doc.sents):
        print(f"S{i}: {s.text}")
    print("-" * 50)
    return clean_text


# =============================================================
# PHASE 1 — CHARACTER INPUT + EXTRACTION
# =============================================================

# Canonical pronoun sets — used to build pronoun→character maps.
# Each entry: (subject, object, possessive_det, possessive_pro, reflexive)
PRONOUN_SETS = {
    "she/her":   {"she", "her", "hers", "herself"},
    "he/him":    {"he", "him", "his", "himself"},
    "they/them": {"they", "them", "their", "theirs", "themselves", "themself"},
}


def _parse_pronoun_key(raw):
    """
    Normalises a student-typed pronoun string into a canonical key.

    Example:
        'She/Her' → 'she/her'
        'he'      → 'he/him'
        'ze/hir'  → 'ze/hir'  (custom, returned as-is)
    """
    raw = raw.strip().lower()
    # Try to match against known sets
    for key in PRONOUN_SETS:
        subject = key.split("/")[0]
        if raw in (key, subject, key.replace("/", " ")):
            return key
    # Custom pronoun — return as-is (student can enter 'ze/hir' etc.)
    return raw


def collect_characters_from_input():
    """
    Interactive input loop (works in Colab / Jupyter / terminal).
    Students enter one character at a time: name + pronouns.
    Returns char_input: dict {name: pronoun_key}

    pronoun_key is one of:
      'she/her', 'he/him', 'they/them', or a custom string like 'ze/hir'
    """
    print("\n" + "="*55)
    print("📖 CHARACTER SETUP")
    print("  Enter each character's name and pronouns.")
    print("  Press Enter with no name when done.")
    print("  Pronoun options: she/her · he/him · they/them · custom")
    print("="*55)

    char_input = {}   # {canonical_name: pronoun_key}

    while True:
        name = input("\nCharacter name (or press Enter to finish): ").strip()
        if not name:
            if not char_input:
                print("  ⚠️  No characters entered. Please add at least one.")
                continue
            break

        pronoun_raw = input(f"Pronouns for {name}: ").strip()
        if not pronoun_raw:
            pronoun_raw = "unknown"

        pronoun_key = _parse_pronoun_key(pronoun_raw)
        char_input[name] = pronoun_key
        print(f"  ✅ Added: {name} ({pronoun_key})")

    print(f"\n  {len(char_input)} character(s) registered.")
    return char_input


def extract_characters(doc, char_input=None):
    """
    Builds character registries from student input + NER.

    Student input is shown in output. NER-only characters are tracked
    internally to prevent their clauses from polluting student characters,
    but never appear in the final summary.

    Example:
        char_input = {"Emma Smith": "she/her", "Leo Smith": "he/him"}
        Story also mentions "Sara Lee" (not in char_input).
        → output_chars = ["Emma Smith", "Leo Smith"]
        → char_pronoun_dict includes Sara Lee internally
        → Sara Lee's clauses won't bleed into Emma/Leo output,
          but Sara Lee herself won't appear in the summary.

    Returns:
      char_pronoun_dict  : all characters including NER extras
      pronoun_to_chars   : e.g. {"she": ["Emma Smith"], "he": ["Leo Smith"]}
      singular_they_chars: e.g. {"V"} if V uses they/them as singular
      output_chars       : ["Emma Smith", "Leo Smith"] — shown in output only
    """
    d = gender.Detector()

    # ── Student-provided characters ───────────────────────────────────────
    char_pronoun_dict = dict(char_input) if char_input else {}
    output_chars = list(char_pronoun_dict.keys())  # only these go to output

    # ── NER: detect additional characters for pollution-prevention only ───
    raw_names = set(ent.text for ent in doc.ents if ent.label_ == "PERSON")
    sorted_names = sorted(raw_names, key=len, reverse=True)
    name_map = {}
    for name in sorted_names:
        if len(name.split()) == 1:
            for full_name in sorted_names:
                if name != full_name and name in full_name.split():
                    name_map[name] = full_name
                    break
        if name not in name_map:
            name_map[name] = name

    student_names_lower = {n.lower() for n in char_pronoun_dict}
    for name in set(name_map.values()):
        if name.lower() not in student_names_lower:
            first = name.split()[0]
            guess = d.get_gender(first.capitalize())
            if 'female' in guess:
                pronoun_key = 'she/her'
            elif 'male' in guess:
                pronoun_key = 'he/him'
            else:
                pronoun_key = 'unknown'
            char_pronoun_dict[name] = pronoun_key
            print(f"  [NER] Extra character detected (internal only): {name}")

    # ── Build pronoun → character lookup ─────────────────────────────────
    pronoun_to_chars = {}
    singular_they_chars = set()

    for name, pkey in char_pronoun_dict.items():
        forms = PRONOUN_SETS.get(pkey, set())
        if not forms:
            parts = pkey.split("/")
            forms = set(parts)
        for form in forms:
            pronoun_to_chars.setdefault(form, []).append(name)
        if pkey == "they/them":
            singular_they_chars.add(name)

    print("\n=== [PHASE 1] Character Registry ===")
    for name in output_chars:
        pkey = char_pronoun_dict[name]
        print(f"  {name:20} | {pkey:12} | source: student")
    print("-" * 50)

    return char_pronoun_dict, pronoun_to_chars, singular_they_chars, output_chars


# =============================================================
# PHASE 2 — COREFERENCE ANNOTATION  (no text replacement)
# =============================================================

# ---------- helpers ----------

REPORTING_VERBS = {
    "said", "whispered", "replied", "asked", "shouted", "cried",
    "murmured", "called", "answered", "stated", "declared", "added",
    "continued", "insisted", "admitted", "exclaimed", "muttered",
    "responded", "argued", "began", "thought", "wondered",
}


def _parse_quote_spans(sent):
    """
    Returns (start_char, end_char) pairs for every quoted span in the sentence.

    Example:
        Sentence: 'Sara said, "I am ready," and Mike said, "Me too."'
        Returns:  [(11, 22), (38, 46)]
        → span 0 covers "I am ready", span 1 covers "Me too"
    """
    spans = []
    in_quote = False
    q_start = None
    for i, ch in enumerate(sent.text):
        if ch == '"':
            if not in_quote:
                in_quote = True
                q_start = i
            else:
                in_quote = False
                spans.append((q_start, i))
    return spans


def _tokens_in_span(sent, char_start, char_end):
    """
    Returns global token indices whose text falls inside [char_start, char_end].

    Example:
        Sentence: 'She said, "I am ready."'
        char_start=10, char_end=22  (the quoted span)
        Returns: {3, 4, 5}  (indices of tokens "I", "am", "ready")
    """
    indices = set()
    for tok in sent:
        rel = tok.idx - sent.start_char
        if char_start < rel < char_end:
            indices.add(tok.i)
    return indices


def _speaker_from_reporting_verb(sent, outside_indices, char_gender_dict):
    """
    Finds the speaker of a quote by locating a reporting verb outside the
    quoted span and walking to its subject.

    Example:
        Sentence: '"I am ready," Sara Lee said.'
        outside_indices contains token for "said"
        "said" → nsubj → "Lee" → matches "Sara Lee"
        Returns: "Sara Lee"
    """
    for tok in sent:
        if tok.i not in outside_indices:
            continue
        if tok.lemma_.lower() in REPORTING_VERBS and tok.pos_ == "VERB":
            for child in tok.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    for name in char_gender_dict:
                        if child.text.lower() in name.lower().split():
                            return name
    return None


def _speaker_from_outside_names(sent, outside_indices, char_gender_dict):
    """
    Fallback speaker detection: returns the first character name token
    that appears outside any quoted span in the sentence.

    Example:
        Sentence: '"I am ready." Sara Lee nodded.'
        No reporting verb found → scans outside tokens → finds "Sara"
        Returns: "Sara Lee"
    """
    for tok in sent:
        if tok.i not in outside_indices:
            continue
        for name in char_gender_dict:
            if tok.text.lower() in name.lower().split():
                return name
    return None


def _find_speakers_per_quote(sent, char_gender_dict, last_speaker):
    """
    Identifies the speaker for each quoted span in a sentence independently,
    so two quotes in one sentence can have different speakers.

    Example:
        Sentence: 'Sara said, "I am ready," and Mike replied, "Me too."'
        Span 0 → adjacent verb "said" → nsubj "Sara" → speaker = Sara Lee
        Span 1 → adjacent verb "replied" → nsubj "Mike" → speaker = Mike Park
        Returns: {0: "Sara Lee", 1: "Mike Park"}

    Three-tier fallback per span:
        1. Reporting verb nsubj adjacent to THIS quote
        2. Any named character outside all quotes in the sentence
        3. last_speaker carried over from the previous sentence
    """
    spans = _parse_quote_spans(sent)
    if not spans:
        return {}

    # All token indices that are inside ANY quote
    all_quote_indices = set()
    for qs, qe in spans:
        all_quote_indices |= _tokens_in_span(sent, qs, qe)

    # Token indices outside all quotes
    outside_indices = {tok.i for tok in sent} - all_quote_indices

    result = {}
    for span_idx, (qs, qe) in enumerate(spans):
        # Try reporting verb that is adjacent to THIS specific quote.
        # "Adjacent" = outside all quotes AND within ±5 tokens of the quote boundary.
        quote_tok_positions = _tokens_in_span(sent, qs, qe)
        if quote_tok_positions:
            min_q = min(quote_tok_positions)
            max_q = max(quote_tok_positions)
            adjacent = {
                tok.i for tok in sent
                if tok.i in outside_indices and (
                    abs(tok.i - min_q) <= 5 or abs(tok.i - max_q) <= 5
                )
            }
        else:
            adjacent = outside_indices

        speaker = _speaker_from_reporting_verb(sent, adjacent, char_gender_dict)

        # Fallback 1: any name outside all quotes in this sentence
        if not speaker:
            speaker = _speaker_from_outside_names(sent, outside_indices, char_gender_dict)

        # Fallback 2: carry forward last known speaker across sentences
        if not speaker:
            speaker = last_speaker

        result[span_idx] = speaker

    return result


def resolve_references(text, char_pronoun_dict, pronoun_to_chars, singular_they_chars):
    """
    Annotates pronouns with their resolved character names WITHOUT modifying
    the original sentences. Builds a coref_map keyed by sentence-relative
    token index so downstream code can look up who any pronoun refers to.

    Example:
        Story: "Emma arrived. She smiled. They left."
        coref_map = {
            (1, 0): "Emma Smith",          # 'She' in S1 → Emma
            (2, 0): ["Emma Smith", "Leo"], # 'They' in S2 → both
        }

    Handles:
        she/her   → Emma Smith (she/her pronoun user)
        he/him    → Leo Smith
        they/them → singular: V (if V is a they/them character alone)
                    plural:   [Emma, Leo] (if group context)
        I (in quotes) → speaker identified via reporting verb
        we/us     → plural group
    """
    # All non-they/them, non-we pronoun forms (looked up directly)
    plural_we       = {"we", "us", "our", "ours"}
    first_person_sg = {"i", "me", "my", "mine"}
    # they/them forms — need special handling (singular vs plural)
    they_forms      = {"they", "them", "their", "theirs", "themselves", "themself"}

    char_list = list(char_pronoun_dict.keys())

    original_doc   = nlp(text)
    original_sents = [s.text for s in original_doc.sents]

    doc_ai = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    _ = doc_ai._.resolved_text

    coref_map         = {}
    last_speaker      = None
    last_plural_group = []
    last_singular_they = None
    # Track which sentence each character was last explicitly named in.
    # Used to decide whether singular-they 'them' (object) resolves to V
    # or to the plural group — whichever was named more recently wins.
    # Example: 'V is sad. Emma watches them leave.'
    #   last_seen_sent[V] = 0, last_seen_sent[Emma] = 1
    #   plural group last seen at max(last_seen_sent[Emma], last_seen_sent[Leo])
    #   V last seen at 0, plural group last seen at 1 → them = plural (Emma)
    # Example: 'Emma and Leo argued. V left quietly. Emma watches them go.'
    #   V last seen at 1, plural group last seen at 0 → them = V
    last_seen_sent    = {}  # {char_name: sent_idx}

    print("\n=== [PHASE 2] Coreference Annotation (original text preserved) ===")

    for sent_idx, sent in enumerate(original_doc.sents):
        s_text     = sent.text
        sent_start = sent.start

        # ── Characters explicitly named in this sentence ──────────────────
        named_in_sent = [
            name for name in char_list
            if name.lower() in s_text.lower()
        ]

        # ── Update last_singular_they and last_seen_sent ──────────────────
        for name in named_in_sent:
            last_seen_sent[name] = sent_idx
            if name in singular_they_chars:
                last_singular_they = name

        # ── Update last_plural_group (non-singular-they chars only) ───────
        plural_named = [n for n in named_in_sent if n not in singular_they_chars]
        if len(plural_named) >= 2:
            for name in plural_named:
                if name not in last_plural_group:
                    last_plural_group.append(name)

        base_plural_target = list(last_plural_group) if last_plural_group else []

        # ── Per-quote speaker resolution ──────────────────────────────────
        has_quote = '"' in s_text
        if has_quote:
            quote_spans     = _parse_quote_spans(sent)
            speaker_by_span = _find_speakers_per_quote(
                sent, char_pronoun_dict, last_speaker
            )
            if speaker_by_span:
                last_val = speaker_by_span.get(len(quote_spans) - 1)
                if last_val:
                    last_speaker = last_val

            token_speaker = {}
            for span_idx, (qs, qe) in enumerate(quote_spans):
                spk = speaker_by_span.get(span_idx)
                if not spk:
                    continue
                for tok in sent:
                    rel     = tok.idx - sent.start_char
                    rel_idx = tok.i - sent_start
                    if qs < rel < qe:
                        token_speaker[rel_idx] = spk
        else:
            token_speaker = {}

        # ── Check if a singular they/them char is the nsubj of this sentence ─
        # Only runs when the story has at least one they/them character (e.g. V).
        # If nobody uses they/them as singular pronouns, skip entirely —
        # all they/them tokens will go straight to plural logic.
        # Example: story has only she/her and he/him characters → skipped.
        # Example: story has V (they/them) → scan for V as nsubj each sentence.
        singular_they_nsubj = None
        if singular_they_chars:
            for tok in sent:
                if tok.dep_ == "nsubj":
                    for c in singular_they_chars:
                        if tok.text.lower() == c.lower().split()[0] or \
                           tok.text.lower() == c.lower():
                            singular_they_nsubj = c
                            break
                if singular_they_nsubj:
                    break

        # ── Print sentence first, then annotate pronouns below it ────────
        print(f"  S{sent_idx}: {s_text}")

        # ── Annotate each token ───────────────────────────────────────────
        for token in sent:
            tok_lower = token.text.lower()
            rel_idx   = token.i - sent_start

            # ── they/them forms: singular vs plural ───────────────────────
            if tok_lower in they_forms:
                token_is_subject = token.dep_ in {
                    "nsubj", "nsubjpass", "csubj", "csubjpass", "expl"
                }

                if token_is_subject:
                    # Subject-role: use nsubj rule (V is this sentence's subject)
                    if singular_they_nsubj:
                        coref_map[(sent_idx, rel_idx)] = singular_they_nsubj
                        print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                              f"→ {singular_they_nsubj} [singular they, nsubj rule]")
                    elif last_singular_they and len(last_plural_group) < 2:
                        coref_map[(sent_idx, rel_idx)] = last_singular_they
                        print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                              f"→ {last_singular_they} [singular they]")
                    else:
                        if not base_plural_target:
                            print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                                  f"→ [skipped: no prior group]")
                            continue
                        solo = named_in_sent[0] if len(named_in_sent) == 1 else None
                        plural_target = (
                            [c for c in base_plural_target if c != solo]
                            if solo else base_plural_target
                        )
                        if plural_target:
                            coref_map[(sent_idx, rel_idx)] = plural_target
                            print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                                  f"→ {plural_target} [plural they]")

                else:
                    # Object-role them/their always resolves to plural group.
                    # Example: 'V watched them' → them = Emma+Leo (plural)
                    # Example: 'their stall was overturned' → their = Emma+Leo
                    if not base_plural_target:
                        print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                              f"→ [skipped: no prior group]")
                        continue
                    solo = named_in_sent[0] if len(named_in_sent) == 1 else None
                    plural_target = (
                        [c for c in base_plural_target if c != solo]
                        if solo else base_plural_target
                    )
                    if plural_target:
                        coref_map[(sent_idx, rel_idx)] = plural_target
                        print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                              f"→ {plural_target} [plural they]")

            # ── we/us/our: always plural group ───────────────────────────
            elif tok_lower in plural_we:
                if not base_plural_target:
                    print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                          f"→ [skipped: no prior group]")
                    continue
                solo = named_in_sent[0] if len(named_in_sent) == 1 else None
                plural_target = (
                    [c for c in base_plural_target if c != solo]
                    if solo else base_plural_target
                )
                if plural_target:
                    coref_map[(sent_idx, rel_idx)] = plural_target
                    print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                          f"→ {plural_target} [we/us]")

            # ── other pronouns: look up via pronoun_to_chars ─────────────
            elif tok_lower in pronoun_to_chars:
                targets = pronoun_to_chars[tok_lower]
                # If multiple characters share the same pronoun form,
                # pick the most recently named one in the story
                if len(targets) == 1:
                    resolved = targets[0]
                else:
                    # Pick whichever target was most recently named
                    resolved = next(
                        (n for n in reversed(named_in_sent) if n in targets),
                        targets[0]
                    )
                coref_map[(sent_idx, rel_idx)] = resolved
                print(f"  S{sent_idx} '{token.text}' [{token.dep_}] "
                      f"(rel {rel_idx}) → {resolved}")

            # ── first-person singular inside quote ────────────────────────
            elif tok_lower in first_person_sg and has_quote:
                spk = token_speaker.get(rel_idx)
                if spk:
                    coref_map[(sent_idx, rel_idx)] = spk
                    print(f"  S{sent_idx} '{token.text}' (rel {rel_idx}) "
                          f"→ {spk} [dialogue]")

    print("-" * 50)
    return original_sents, coref_map


# =============================================================
# PHASE 3 — CHARACTER-CLAUSE ATTRIBUTION
# =============================================================

def _get_subtree_text(token, excluded_indices=None):
    """
    Returns the text of a token's subtree, skipping excluded token indices.
    Also removes stranded punctuation and conjunctions left over after exclusion.

    Example:
        Subtree of ROOT 'consumed': [While, Emma, celebrated, , Leo, was, consumed, ...]
        excluded_indices = {indices of 'While Emma celebrated' advcl}
        Naive result: ', Leo was consumed by fear'  ← stranded comma
        After cleanup: 'Leo was consumed by fear'   ← comma removed
    """

    # Also exclude punct/cc tokens that are stranded after exclusion:
    # if a comma or conjunction immediately precedes or follows only
    # excluded tokens, it should be excluded too.
    tokens = [t for t in token.subtree]
    final_excluded = set(excluded_indices)

    for i, t in enumerate(tokens):
        if t.dep_ in ("punct", "cc") and t.i not in final_excluded:
            # Check if all neighbours (prev and next non-punct) are excluded
            prev_content = next(
                (tok for tok in reversed(tokens[:i])
                 if tok.dep_ not in ("punct", "cc")), None
            )
            next_content = next(
                (tok for tok in tokens[i+1:]
                 if tok.dep_ not in ("punct", "cc")), None
            )
            prev_excluded = (prev_content is None or prev_content.i in final_excluded)
            next_excluded = (next_content is None or next_content.i in final_excluded)
            if prev_excluded or next_excluded:
                final_excluded.add(t.i)

    return " ".join(t.text for t in tokens if t.i not in final_excluded)


def _name_in_subtree(token, char_list):
    """
    Returns which characters from char_list are explicitly named in
    token's subtree. Uses NER for multi-word names to avoid surname collisions.

    Example:
        Subtree of 'celebrated': [Emma, Smith, celebrated, her, courage]
        char_list = ["Emma Smith", "Leo Smith"]
        NER finds span "Emma Smith" whose root 'Smith' is in subtree
        → returns {"Emma Smith"}   (not "Leo Smith", even though 'Smith' appears)

        For single-word name 'V':
        Subtree contains token 'V' → direct text match → returns {"V"}
    """
    result = set()
    doc = token.doc
    subtree_ids = {t.i for t in token.subtree}

    single_word = [c for c in char_list if len(c.split()) == 1]
    multi_word  = [c for c in char_list if len(c.split()) > 1]

    # Single-word: direct match
    for t in token.subtree:
        for char in single_word:
            if t.text.lower() == char.lower():
                result.add(char)

    # Multi-word: NER span root must be in subtree
    for ent in doc.ents:
        if ent.label_ != "PERSON" or ent.root.i not in subtree_ids:
            continue
        ent_lower = ent.text.lower()
        for char in multi_word:
            if ent_lower in char.lower() or char.lower() in ent_lower:
                result.add(char)
                break

    return result


def _chars_in_subtree(token, char_list):
    """
    Name-only version of character detection in a subtree (no pronoun resolution).

    Example:
        Used to check if 'celebrated' subtree contains Emma Smith before
        deciding whether to prune it from Leo's clause.
    """
    return list(_name_in_subtree(token, char_list))


def _chars_in_subtree_coref(token, char_list, coref_map, sent_idx):
    """
    Finds characters in a subtree by name OR resolved pronoun.
    Used in exclusion logic where the subtree subject may be a pronoun.

    Example:
        Subtree of advcl 'was': [Because, she, was, tired]
        'she' is not a name, but coref_map[(sent_idx, she.i)] = "Emma Smith"
        → returns ["Emma Smith"]
        This lets Leo's clause correctly prune the advcl as belonging to Emma.
    """
    result = _name_in_subtree(token, char_list)

    # Coref map: pronouns resolved to a character
    for t in token.subtree:
        resolved = coref_map.get((sent_idx, t.i))
        if resolved is not None:
            targets = resolved if isinstance(resolved, list) else [resolved]
            for tgt in targets:
                if tgt in char_list:
                    result.add(tgt)

    return list(result)


def _conj_has_own_subject(token):
    """
    Returns True if a conj verb node has its own explicit nsubj child,
    indicating it is an independent conjoined clause rather than a
    shared-subject coordination.

    Example:
        'Emma laughed and Leo cried.'
        → 'cried' (conj) has nsubj 'Leo' → returns True → stop climbing here
           Leo gets 'Leo cried', not the full sentence.

        'Emma and Leo walked.'
        → 'Leo' (conj) is a noun, not a verb, has no nsubj → returns False
           Both characters climb to ROOT 'walked' and get the full sentence.
    """
    return any(child.dep_ == "nsubj" for child in token.children)


def _climb_to_head(anchor_token):
    """
    Climbs the dependency tree with three stop conditions:

    1. Standard stops: ROOT, advcl, ccomp, relcl.

    2. conj with its own nsubj → independent conjoined clause, stop here.
       - 'Emma laughed and Leo cried':
           'cried' is conj with own nsubj 'Leo' → stop at 'cried'
           Leo gets 'Leo cried', not the full sentence.
       - 'Emma and Leo walked':
           'Leo' is conj of noun 'Emma', not a verb with own nsubj
           → keeps climbing to ROOT 'walked'
           Both characters get the full sentence.

    3. xcomp whose parent has an obj matching the anchor token →
       the anchor is the implicit subject of the xcomp, stop here.
       Covers all three patterns:
         - 'He watches her walk away'  → her = obj, walk = xcomp
         - 'She made him leave'        → him = obj, leave = xcomp
         - 'He heard her singing'      → her = obj, singing = xcomp
       In all cases the obj-character gets the xcomp subtree,
       not the full sentence.
    """
    head = anchor_token
    while head.head != head:
        if head.dep_ in ("ROOT", "advcl", "ccomp", "relcl"):
            break
        if head.dep_ == "conj" and _conj_has_own_subject(head):
            break
        # xcomp stop: anchor is the obj of xcomp's parent verb
        if head.dep_ == "xcomp":
            parent_objs = {c for c in head.head.children if c.dep_ == "obj"}
            if anchor_token in parent_objs:
                break
        head = head.head
    return head


def _ccomp_nsubj(token):
    """
    Returns the nsubj child of a ccomp verb node, or None.
    Used to decide whether the embedded clause belongs to the current
    character or someone else.

    Example:
        'Emma told Leo that he was wrong.'
        ccomp node = 'was'
        → nsubj child = 'he' → returns token 'he'
        → 'he' resolves to Leo → ccomp belongs to Leo, not Emma
    """
    for child in token.children:
        if child.dep_ == "nsubj":
            return child
    return None


def _token_resolves_to(token, char, coref_map, sent_idx):
    """
    Returns True if a token refers to a given character, either by direct
    name match or via the coref_map pronoun resolution.

    Example:
        token = 'she', char = 'Emma Smith'
        coref_map[(sent_idx, she.i)] = 'Emma Smith'
        → returns True

        token = 'Smith', char = 'Emma Smith'
        'smith' in ['emma', 'smith'] → returns True (direct name match)

        token = 'he', char = 'Emma Smith'
        coref_map[(sent_idx, he.i)] = 'Leo Smith'
        → returns False
    """
    # Direct name match
    if token.text.lower() in [p.lower() for p in char.split()]:
        return True
    # Coref map match (token.i is already sentence-relative in attribute_clauses)
    resolved = coref_map.get((sent_idx, token.i))
    if resolved is None:
        return False
    targets = resolved if isinstance(resolved, list) else [resolved]
    return char in targets


def _extract_clause_for_anchor(anchor_token, found_chars, char,
                                coref_map=None, sent_idx=None):
    """
    Climbs to the governing clause head for an anchor token, then prunes
    sub-clauses belonging exclusively to other characters.

    Example A — advcl separation:
        'While Emma celebrated, Leo was consumed by fear.'
        Emma's anchor 'Smith' → stops at advcl 'celebrated'
        → Emma gets 'While Emma celebrated her courage'
        Leo's anchor 'Smith' → climbs to ROOT 'consumed'
        → exclusion prunes advcl (contains Emma, not Leo)
        → Leo gets 'Leo was consumed by terrible fear'

    Example B — split-subject conj:
        'Emma laughed and Leo cried.'
        Emma's anchor → ROOT 'laughed', exclusion prunes conj 'cried' (Leo's)
        → Emma gets 'Emma laughed'
        Leo's anchor → stops at conj 'cried' (has own nsubj)
        → Leo gets 'Leo cried'

    Example C — self-referential ccomp (skip):
        'Emma believed that she would succeed.'
        'she' → ccomp 'succeed' → nsubj 'she' resolves to Emma → skip
        Emma's name anchor → ROOT 'believed' → gets full sentence
    """
    if coref_map is None:
        coref_map = {}

    other_chars = set(found_chars) - {char}
    head = _climb_to_head(anchor_token)

    # ── ccomp special handling ────────────────────────────────────────────
    if head.dep_ == "ccomp":
        nsubj = _ccomp_nsubj(head)
        if nsubj is not None:
            subj_is_self = _token_resolves_to(nsubj, char, coref_map, sent_idx)
            if subj_is_self:
                # e.g. "Emma believed that she would succeed"
                # The anchor pronoun "she" climbed to ccomp "succeed",
                # but its nsubj refers back to Emma herself.
                # Skip — the ROOT clause (believed that she would succeed)
                # already captures this for Emma.
                return head, ""
        # Otherwise the ccomp belongs to another character → keep it
        clause_text = _get_subtree_text(head).strip()
        return head, clause_text

    # ── ROOT / conj / advcl: apply coref-aware exclusion logic ──────────
    # Prune any child clause that belongs exclusively to OTHER characters.
    # Prunable dep_ types:
    #   advcl  — all adverbial clause subtypes (when/because/if/although...)
    #   ccomp  — complement clauses
    #   relcl  — relative clauses
    #   conj   — split-subject conjoined clauses (Emma laughed AND Leo cried)
    #            only prune conj if it has its own nsubj (independent clause)
    excluded = set()
    if head.dep_ in ("ROOT", "conj", "advcl"):
        for child in head.children:
            prune_candidate = child.dep_ in ("advcl", "ccomp", "relcl")
            # Only prune conj if it's an independent clause (has own nsubj)
            if child.dep_ == "conj" and _conj_has_own_subject(child):
                prune_candidate = True

            if prune_candidate:
                child_others  = _chars_in_subtree_coref(
                    child, list(other_chars), coref_map, sent_idx
                )
                child_current = _chars_in_subtree_coref(
                    child, [char], coref_map, sent_idx
                )
                if child_others and not child_current:
                    excluded.update(t.i for t in child.subtree)

    clause_text = _get_subtree_text(head, excluded).strip()
    return head, clause_text


def _get_root_token(doc):
    """
    Returns the ROOT token of a parsed sentence, or None.

    Example:
        'While Emma celebrated, Leo was consumed by fear.'
        → returns token 'consumed' (dep_ == 'ROOT')
    """
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None


def _root_is_covered(root_tok, attribution, sent_idx, char_list):
    """
    Returns True if any character already has a clause rooted at the
    sentence ROOT for this sentence index.

    Used to detect the special case where ALL character anchors are in
    advcl branches and nobody has reached ROOT yet — in that case the
    main clause content would be lost without intervention.

    Example:
        'When Emma laughed, because Leo was nervous, the room fell silent.'
        After processing: Emma → advcl 'When Emma laughed'
                          Leo  → advcl 'because Leo was nervous'
        ROOT 'fell' has no entry → _root_is_covered returns False
        → Rule 4 triggers: merge ROOT text into each character's advcl
    """
    for char in char_list:
        for entry in attribution[char]:
            if entry["idx"] == sent_idx and entry["head_i"] == root_tok.i:
                return True
    return False


def attribute_clauses(original_sentences, char_list, coref_map, output_chars=None):
    """
    Assigns each sentence's clauses to the characters they belong to,
    using dependency parsing and the coref_map for pronoun resolution.

    char_list    : ALL characters including NER extras — ensures their
                   clauses are accounted for in exclusion logic.
    output_chars : only student-provided characters — only these appear
                   in the printed summary and return value.

    Example:
        S2: 'While Emma celebrated, Leo was consumed by fear.'
        Emma anchor → advcl → clause: 'While Emma celebrated her courage'
        Leo anchor  → ROOT  → exclusion removes advcl → clause: 'Leo was consumed by fear'
        Sara (NER-only, not in output_chars) also mentioned → tracked internally
        but Sara's entry is excluded from the printed summary.
    """
    if output_chars is None:
        output_chars = char_list
    MODIFIER_DEPS = {"poss", "det", "amod", "nmod", "nummod", "quantmod"}

    print("\n=== [PHASE 3] Character-Clause Attribution ===")
    attribution = {c: [] for c in char_list}

    for sent_idx, sent_text in enumerate(original_sentences):
        doc = nlp(sent_text)
        root_tok = _get_root_token(doc)

        # ── Build anchor token lists ──────────────────────────────────────
        # CRITICAL: for each character, we want exactly ONE anchor token
        # per name occurrence — the syntactic head of the NER span.
        # Using every word in the name as separate anchors causes duplicate
        # clause extraction (e.g. both "Emma" and "Smith" as anchors).
        char_tokens = {c: [] for c in char_list}

        # Primary: use NER spans — pick the syntactic root of each span
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            ent_lower = ent.text.lower()
            for char in char_list:
                char_lower = char.lower()
                if ent_lower in char_lower or char_lower in ent_lower:
                    anchor = ent.root
                    if anchor not in char_tokens[char]:
                        char_tokens[char].append(anchor)
                    break

        # Fallback: if NER missed a character, match by first name token only
        # (avoids the multi-token duplicate problem)
        for char in char_list:
            if char_tokens[char]:
                continue  # already found via NER
            first_name = char.split()[0].lower()
            for token in doc:
                if token.text.lower() == first_name:
                    if token not in char_tokens[char]:
                        char_tokens[char].append(token)

        # Coref-map pronouns: add as anchors ONLY if they are in a subject role.
        # Object-role pronouns (dobj, iobj, pobj) mean the character is being
        # acted upon, not acting — they should not generate clause attribution.
        SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
        for token in doc:
            resolved = coref_map.get((sent_idx, token.i))
            if resolved is not None:
                # Only add if this token is in a subject role
                if token.dep_ not in SUBJECT_DEPS:
                    continue
                targets = resolved if isinstance(resolved, list) else [resolved]
                for tgt in targets:
                    if tgt in char_tokens and token not in char_tokens[tgt]:
                        char_tokens[tgt].append(token)

        found_chars = [c for c, toks in char_tokens.items() if toks]

        # Deduplicate anchors per character
        for char in found_chars:
            seen_anchor_heads = set()
            deduped = []
            for tok in char_tokens[char]:
                h = _climb_to_head(tok)
                if h.i not in seen_anchor_heads:
                    seen_anchor_heads.add(h.i)
                    deduped.append(tok)
            char_tokens[char] = deduped



        # ── Extract & deduplicate clauses ─────────────────────────────────
        for char in found_chars:
            seen_heads = set()

            # Determine if all non-modifier anchors live inside a conj branch
            # (not the ROOT's direct subject). Used to block incorrect ROOT claims.
            non_mod_anchors = [
                a for a in char_tokens[char] if a.dep_ not in MODIFIER_DEPS
            ]
            all_anchors_in_conj_branch = (
                len(non_mod_anchors) > 0 and root_tok is not None and
                all(
                    any(
                        anc.dep_ == "conj" and anc.head.i == root_tok.i
                        for anc in list(a.ancestors)
                    )
                    for a in non_mod_anchors
                )
            )

            for anchor in char_tokens[char]:
                # Skip possessives and other modifier roles
                if anchor.dep_ in MODIFIER_DEPS:
                    continue

                head, clause_text = _extract_clause_for_anchor(
                    anchor, found_chars, char,
                    coref_map=coref_map, sent_idx=sent_idx
                )
                if not clause_text:
                    continue

                # Block ROOT claim when char only exists in a conj branch
                if (root_tok is not None and head.i == root_tok.i
                        and all_anchors_in_conj_branch):
                    continue

                if head.i not in seen_heads:
                    seen_heads.add(head.i)
                    attribution[char].append({
                        "idx":    sent_idx,
                        "head_i": head.i,
                        "text":   clause_text,
                    })

        # ── Rule 4: ROOT uncovered — merge main clause into advcl chars ───
        if root_tok and not _root_is_covered(root_tok, attribution, sent_idx, char_list):
            for char in found_chars:
                char_entries = [e for e in attribution[char] if e["idx"] == sent_idx]
                if not char_entries:
                    continue
                other_chars = set(found_chars) - {char}
                excluded = set()
                for child in root_tok.children:
                    if child.dep_ in ("advcl", "relcl"):
                        child_others  = _chars_in_subtree_coref(
                            child, list(other_chars), coref_map, sent_idx)
                        child_current = _chars_in_subtree_coref(
                            child, [char], coref_map, sent_idx)
                        if child_others and not child_current:
                            excluded.update(t.i for t in child.subtree)
                root_text = _get_subtree_text(root_tok, excluded).strip()
                if not root_text:
                    continue
                already_root = any(
                    e["idx"] == sent_idx and e["head_i"] == root_tok.i
                    for e in attribution[char]
                )
                if not already_root:
                    advcl_texts = " ".join(e["text"] for e in char_entries)
                    merged_text = f"{advcl_texts} {root_text}".strip()
                    attribution[char] = [
                        e for e in attribution[char]
                        if not (e["idx"] == sent_idx and e["head_i"] != root_tok.i)
                    ]
                    attribution[char].append({
                        "idx":    sent_idx,
                        "head_i": root_tok.i,
                        "text":   merged_text,
                    })

        # ── Print per-sentence results ────────────────────────────────────
        print(f"\n  S{sent_idx}: {sent_text}")
        for char in output_chars:
            sent_entries = [e for e in attribution.get(char, []) if e["idx"] == sent_idx]
            if sent_entries:
                for e in sent_entries:
                    print(f"    → {char}: \"{e['text']}\"")

    # ── Print summary (only student-provided characters) ──────────────────
    print("\n📊 FINAL ATTRIBUTION SUMMARY (By Entity):")
    for char in output_chars:
        entries = attribution.get(char, [])
        print(f"\n👤 {char.upper()}")
        print("-" * 40)
        if not entries:
            print("  (No clauses attributed)")
        for e in entries:
            print(f"  [S{e['idx']}] {e['text']}")
    print("\n" + "=" * 50)

    return {c: attribution[c] for c in output_chars}


# =============================================================
# MASTER PIPELINE
# =============================================================
def run_full_pipeline(text, char_input=None):
    """
    Runs all four phases in sequence and returns the attribution results.
    Every phase is individually protected — if any phase fails, it prints
    a warning and returns safe empty defaults so downstream code still runs.

    Example:
        char_input = {"Emma Smith": "she/her", "Leo Smith": "he/him"}
        text = "Emma arrived. She smiled. Leo waited."

        Phase 0: splits into S0, S1, S2
        Phase 1: registers Emma Smith + Leo Smith; NER may find extras
        Phase 2: 'She' → Emma Smith in coref_map
        Phase 3: each sentence's clauses attributed to their character
        Returns: (["Emma Smith", "Leo Smith"], original_sents, coref_map, attribution)
    """
    print("🚀 Starting DH Pipeline...\n")

    # ── Phase 0+1: preprocessing and character extraction ────────────────
    try:
        clean_text = preprocess_text(text)
    except Exception as e:
        print(f"⚠️  Phase 0 error: {e}\n    Using raw text as fallback.")
        clean_text = text.strip()

    try:
        doc_init = nlp(clean_text)
        char_pronoun_dict, pronoun_to_chars, singular_they_chars, output_chars = \
            extract_characters(doc_init, char_input=char_input)
        char_list = list(char_pronoun_dict.keys())
    except Exception as e:
        print(f"⚠️  Phase 1 error: {e}\n    Using char_input as-is.")
        char_pronoun_dict = dict(char_input) if char_input else {}
        output_chars      = list(char_pronoun_dict.keys())
        char_list         = output_chars
        pronoun_to_chars  = {}
        singular_they_chars = set()

    # ── Phase 2: coreference annotation ──────────────────────────────────
    try:
        original_sents, coref_map = resolve_references(
            clean_text, char_pronoun_dict, pronoun_to_chars, singular_they_chars
        )
    except Exception as e:
        print(f"⚠️  Phase 2 error: {e}\n    Skipping coreference — pronouns will not be resolved.")
        import re as _re
        original_sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', clean_text) if s.strip()]
        coref_map = {}

    # ── Phase 3: clause attribution ───────────────────────────────────────
    try:
        final_data = attribute_clauses(
            original_sents, char_list, coref_map, output_chars=output_chars
        )
    except Exception as e:
        print(f"⚠️  Phase 3 error: {e}\n    Returning empty attribution.")
        final_data = {c: [] for c in output_chars}

    print("\n✅ All modular analysis complete.")
    return output_chars, original_sents, coref_map, final_data


