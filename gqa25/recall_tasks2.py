# -*- coding: utf-8 -*-
"""recall_tasks2 -- eight more retrieval families, in natural FORMATS rather than random symbols.

WHY.  task_lib's seven synthetic tasks plant fresh random token ids.  Token-wise every example is
distinct, but each task is ONE fixed layout ([k v k v ...]) and a model can learn seven layouts as
seven special cases.  factgen is the only natural-language family.  These eight add the shapes real
text stores facts in -- chat logs, records in six serialisations, code constants in six languages,
schedules, inventories, glossaries, object moves, quotations -- each built combinatorially so that a
family has >= 10,000 distinct STRUCTURES (template x format x field x phrasing, values masked), not
just distinct values.  measure_distinct() below counts both.

CONTRACT (identical to task_lib's generators): gen(w, T, rng, cue, want_mask) -> window of len(w) [,
float32 answer mask].  The answer mask marks ONLY the tokens that must be recalled; the given parts of
a query (the key, the field name) are 0.  `cue` (the trainer's "repeat what I said" strings) is
ignored: every family here carries its own natural query.  If the window cannot hold the task the
window is returned unchanged with mask None, which compose() treats as "nothing planted".

DISJOINT FROM THE RECALL BATTERY (battery3.py), so its cells stay a test of transfer:
  * no battery frame: never "access code for", "password for", "'s access code", "'s password"
    (asserted on every rendered example by _check_disjoint)
  * no battery word-pair vocabulary (silver crimson hollow ... / falcon lantern harbor ...)
  * the battery's names may appear: a name is context, never an answer there.

Registered into task_lib.GENERATORS on import (additive; task_lib itself is not edited, so the
live run's trainer, which never imports this module, is untouched).
"""
import numpy as np
import factgen2 as F2
import task_lib as TL
from task_lib import _splice, _gap, _ret
import tiktoken

enc = tiktoken.get_encoding("gpt2")
_E = enc.encode_ordinary

# ---------------------------------------------------------------- shared pools
_BATTERY_WORDS = set("silver crimson hollow northern quiet amber distant velvet "
                     "falcon lantern harbor meadow cipher anchor thistle beacon".split())
_BATTERY_FRAMES = ("access code for", "password for", "'s access code", "'s password")

FIRST = ("Maria James Aisha Noah Priya Omar Lena Tariq Sofia Diego Yuki Kwame Ingrid Raj Elena Mateo "
         "Fatima Hugo Nadia Kira Wesley Anders Rosa Linh Bilal Marta Soren Ayana Felix Noor Sven Amara "
         "Bjorn Chiara Esme Hassan Iris Jamal Liam Mira Nikolai Olga Quinn Sami Tessa Umar Vera Yara Boris "
         "Cleo Dario Enzo Faye Gino Ivo Juno Kofi Lucia Malik Nell Otto Pia Remy Saoirse Theo Uma Vik "
         "Wren Ximena Yusuf Zora Arlo Bea Cyrus Dina Elio Freya Gus Hedda Idris Jonas Kaia Leif").split()
LAST = ("Okafor Lindqvist Moreau Tanaka Haddad Novak Brennan Castillo Ferreira Ivanova Kowalski Mensah "
        "Nakamura Oyelaran Petrov Quintero Rasmussen Silva Thorne Underwood Varga Whitlock Xu Yilmaz "
        "Zielinski Abara Bhatt Costa Duval Engel").split()
CHAT_HANDLES = ("mira_k jdawson pixelfox tessa.r omar_h quinnq noorbot k_ingrid felixw linh.dev "
                "rosa_m bilal99 soren_t ayana.j wes_ley cleo.v dario_b faye_x gino.p juno_r").split()

def _person(r):
    return F2._pick(r, FIRST) + ("" if r.random() < 0.6 else " " + F2._pick(r, LAST))

# value generators: (kind, gen).  Every one draws fresh per example; word pairs use factgen2's
# battery-disjoint PADJ x PNOUN.
_WORDS = [x for x in F2.COLORS + F2.MATERIALS + F2.ANIMALS + F2.LANGS + F2.PLACES + F2.JOBS
          if x not in _BATTERY_WORDS]     # factgen2.COLORS holds "silver", a battery adjective
_DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
_MONTHS = "January February March April May June July August September October November December".split()
VALUES = {
    "num":    lambda r: F2._num(r, 2, 9999),
    "small":  lambda r: F2._num(r, 2, 99),
    "pair":   F2._phrase,
    "code":   F2._code,
    "word":   lambda r: F2._pick(r, _WORDS),
    "city":   lambda r: F2._pick(r, F2.CITIES),
    "time":   lambda r: "%d:%02d" % (int(r.integers(6, 23)), int(r.choice([0, 5, 10, 15, 20, 30, 40, 45, 50]))),
    "day":    lambda r: F2._pick(r, _DAYS),
    "date":   lambda r: "%d %s" % (int(r.integers(1, 29)), F2._pick(r, _MONTHS)),
    "money":  lambda r: "%d.%02d" % (int(r.integers(1, 900)), int(r.integers(0, 100))),
    "person": lambda r: F2._pick(r, FIRST),
    "hexid":  lambda r: "%x" % int(r.integers(0x1000, 0xfffff)),
}

# ---------------------------------------------------------------- assembly
def _plant(w, rng, facts_txt, queries, skeleton):
    """facts_txt: one string holding every planted fact (its own layout).  queries: list of
    (prefix, answer, suffix) -- prefix and suffix are given, answer is recalled.  Returns
    (window, mask, skeleton) or (None, None, skeleton) when it does not fit."""
    facts = np.array(_E(facts_txt), dtype=np.int64)
    qt, qm = [], []
    for pre, ans, suf in queries:
        a, b = _E(pre), _E(ans)
        c = _E(suf) if suf else []
        qt += a + b + c
        qm += [0.0] * len(a) + [1.0] * len(b) + [0.0] * len(c)
    q = np.array(qt, dtype=np.int64)
    n = len(w)
    room = n - len(facts) - len(q) - 32
    if room < 16:
        return None, None, skeleton
    gap = _gap(rng, 16, room)
    out, m = _splice(w, facts, q, gap, np.empty(0, dtype=np.int64), np.array(qm, dtype=np.float32))
    return out, m, skeleton

def _shrink(build, w, rng, k_max, k_min=1):
    """Call build(k) with the largest k that fits the window, halving on failure."""
    k = k_max
    while k >= k_min:
        out = build(k)
        if out[0] is not None:
            return out
        k = k // 2 if k > 2 else k - 1
    return None, None, None

def _ask(rng, n, frac=(0.4, 1.0)):
    """Which of n planted facts get queried, in a random order (never the planting order)."""
    k = max(1, int(round(n * rng.uniform(*frac))))
    return [int(i) for i in rng.permutation(n)[:k]]

# ================================================================ 1. chat log
_CHAT_THINGS = [  # (thing, value kind)
    ("gate code", "num"), ("locker number", "small"), ("room number", "num"), ("table booking", "time"),
    ("meeting time", "time"), ("train platform", "small"), ("tracking number", "code"), ("printer pin", "num"),
    ("spare key spot", "word"), ("dinner place", "city"), ("flight number", "code"), ("badge number", "num"),
    ("parking spot", "code"), ("shelf label", "pair"), ("batch number", "code"), ("delivery day", "day"),
    ("deadline", "date"), ("budget", "money"), ("team name", "pair"), ("safe word", "pair"),
    ("ticket number", "num"), ("bus route", "small"), ("cabin number", "num"), ("project codename", "pair"),
    ("server name", "pair"), ("invoice total", "money"), ("seat number", "code"), ("handover date", "date"),
    ("contact person", "person"), ("backup day", "day"), ("lab bench", "small"), ("vault row", "code"),
    ("pickup time", "time"), ("venue city", "city"), ("order id", "hexid"), ("router label", "hexid"),
]
_CHAT_SAY = [   # how the fact is first stated -- {t} thing, {v} value
    "the {t} is {v}", "fyi the {t} is {v}", "ok so the {t} is {v} now", "{t}: {v}",
    "just so everyone knows, the {t} is {v}", "the {t} changed, it is {v}", "reminder that the {t} is {v}",
    "{t} = {v}", "noting it here: {t} {v}", "they set the {t} to {v}", "pinning this: the {t} is {v}",
    "for the record the {t} is {v}", "new {t} is {v}", "heads up, {t} is {v}",
]
_CHAT_ASK = [   # (question by the other speaker, the answerer's mid-sentence lead-in)
    ("what was the {t} again?", "the {t} is"), ("remind me of the {t}?", "the {t} is"),
    ("which {t} did we get?", "it is"), ("do you still have the {t}?", "yes, the {t} is"),
    ("sorry, lost the {t}", "no worries, the {t} is"), ("{t}?", "{t}:"), ("what's the {t}", "it's"),
    ("can you repeat the {t}", "sure, the {t} is"),
]
_CHAT_STYLE = [  # (line template with {who} and {msg}, separator before every line)
    ("[{hh}:{mm}] {who}: {msg}", "\n"), ("{who}: {msg}", "\n"), ("<{who}> {msg}", "\n"),
    ("{who} ({hh}:{mm}): {msg}", "\n"), ("- {who}: {msg}", "\n"),
]
_SMALLTALK = ["morning all", "brb", "anyone around?", "thanks!", "on my way", "coffee first",
              "lol", "makes sense", "will check later", "ok", "sounds good", "running late, sorry",
              "can someone review my thing", "nice", "got it", "see you there"]

def chat_recall(w, T, rng, cue=None, want_mask=False):
    style, sep = _CHAT_STYLE[int(rng.integers(0, len(_CHAT_STYLE)))]
    handles = CHAT_HANDLES if rng.random() < 0.5 else FIRST
    def line(who, msg, hh, mm):
        return sep + style.format(who=who, msg=msg, hh="%02d" % hh, mm="%02d" % mm)
    def build(k):
        r = rng
        things = [_CHAT_THINGS[int(i)] for i in r.permutation(len(_CHAT_THINGS))[:k]]
        people = [F2._pick(r, handles) for _ in range(k)]
        vals = [VALUES[kind](r) for _, kind in things]
        hh, mm = int(r.integers(7, 20)), int(r.integers(0, 50))
        lines, sk = [], [style]
        for i, ((t, _), p, v) in enumerate(zip(things, people, vals)):
            if r.random() < 0.4:
                lines.append(line(F2._pick(r, handles), F2._pick(r, _SMALLTALK), hh, mm)); mm = min(59, mm + 1)
            si = int(r.integers(0, len(_CHAT_SAY)))
            lines.append(line(p, _CHAT_SAY[si].format(t=t, v=v), hh, mm)); mm = min(59, mm + int(r.integers(0, 3)))
            sk.append("%s|%d" % (t, si))
        facts = "".join(lines) + "\n"
        qs = []
        hh2 = min(23, hh + int(r.integers(0, 4)))
        for i in _ask(r, k):
            t, _ = things[i]
            qi = int(r.integers(0, len(_CHAT_ASK)))
            asker = F2._pick(r, [h for h in handles if h != people[i]])
            qtxt, lead = _CHAT_ASK[qi]
            pre = (line(asker, ("@%s " % people[i] if r.random() < 0.5 else "") + qtxt.format(t=t), hh2, 0)
                   + line(people[i], lead.format(t=t), hh2, 1))
            qs.append((pre, " " + vals[i], ""))
            sk.append("q%d" % qi)
        return _plant(w, r, facts, qs, "chat:" + "/".join(sk))
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 7)))
    return _finish(w, out, m, sk, want_mask)

# ================================================================ 2. records in six serialisations
_FIELDS = [  # (field name, value kind)
    ("owner", "person"), ("colour", "word"), ("city", "city"), ("status", "word"), ("priority", "small"),
    ("weight", "num"), ("serial", "code"), ("model", "pair"), ("grade", "small"), ("region", "city"),
    ("label", "pair"), ("count", "num"), ("due", "date"), ("price", "money"), ("shift", "day"),
    ("contact", "person"), ("start", "time"), ("checksum", "hexid"), ("batch", "code"), ("tier", "small"),
    ("supplier", "person"), ("depot", "city"), ("finish", "word"), ("ref", "hexid"), ("lot", "code"),
    ("capacity", "num"), ("motto", "pair"), ("slot", "small"), ("origin", "city"), ("assignee", "person"),
    ("deadline", "date"), ("rate", "money"), ("material", "word"), ("alias", "pair"), ("rev", "small"),
    ("host", "pair"), ("zone", "code"), ("budget", "money"), ("opened", "date"), ("closes", "time"),
]
_REC_PREFIX = ["R", "ID", "rec", "item", "unit", "acct", "case", "node", "asset", "job", "ord", "ref"]

def _rec_id(r, pre):
    return "%s-%d" % (pre, int(r.integers(10, 9999)))

def _ser(fmt, rid, kv, idkey):
    """One record in format `fmt`.  kv: list of (field, value)."""
    if fmt == "json":
        return "{" + ", ".join(['"%s": "%s"' % (idkey, rid)] + ['"%s": "%s"' % (f, v) for f, v in kv]) + "}"
    if fmt == "yaml":
        return "%s:\n" % rid + "\n".join("  %s: %s" % (f, v) for f, v in kv)
    if fmt == "pipe":
        return "%s | " % rid + " | ".join("%s: %s" % (f, v) for f, v in kv)
    if fmt == "ini":
        return "[%s]\n" % rid + "\n".join("%s = %s" % (f, v) for f, v in kv)
    if fmt == "pydict":
        return "%s = {" % rid.replace("-", "_") + ", ".join("'%s': '%s'" % (f, v) for f, v in kv) + "}"
    if fmt == "sql":
        return ("INSERT INTO t (%s, %s) VALUES ('%s', %s);"
                % (idkey, ", ".join(f for f, _ in kv), rid, ", ".join("'%s'" % v for _, v in kv)))
    raise ValueError(fmt)

def _rec_query(fmt, rid, f, idkey, r):
    """(prefix, suffix) whose prefix ends exactly where the value starts in that format."""
    if fmt == "json":
        return ('\n{"%s": "%s", "%s": "' % (idkey, rid, f)), '"}'
    if fmt == "yaml":
        return ("\n%s:\n  %s:" % (rid, f)), ""
    if fmt == "pipe":
        return ("\n%s | %s:" % (rid, f)), ""
    if fmt == "ini":
        return ("\n[%s]\n%s =" % (rid, f)), ""
    if fmt == "pydict":
        v = rid.replace("-", "_")
        return ("\n>>> %s['%s']\n'" % (v, f)), "'"
    if fmt == "sql":
        return ("\nSELECT %s FROM t WHERE %s = '%s';\n-- '" % (f, idkey, rid)), "'"
    raise ValueError(fmt)

_REC_FMTS = ["json", "yaml", "pipe", "ini", "pydict", "sql"]
_NOSPACE = {"json", "pydict", "sql"}       # formats whose value follows a quote, not a space

def record_recall(w, T, rng, cue=None, want_mask=False):
    fmt = _REC_FMTS[int(rng.integers(0, len(_REC_FMTS)))]
    pre = F2._pick(rng, _REC_PREFIX)
    idkey = F2._pick(rng, ["id", "key", "name", "code"])
    def build(k):
        r = rng
        nf = int(r.integers(2, 5))
        fields = [_FIELDS[int(i)] for i in r.permutation(len(_FIELDS))[:nf]]   # SHARED by all records
        ids = []
        while len(ids) < k:
            x = _rec_id(r, pre)
            if x not in ids: ids.append(x)
        recs = [[(f, VALUES[kind](r)) for f, kind in fields] for _ in range(k)]
        facts = "\n" + "\n".join(_ser(fmt, rid, kv, idkey) for rid, kv in zip(ids, recs)) + "\n"
        qs, sk = [], ["rec:%s:%s:%s:%s" % (fmt, pre, idkey, ",".join(f for f, _ in fields))]
        for i in _ask(r, k):
            fi = int(r.integers(0, nf))
            f, v = recs[i][fi]
            p, s = _rec_query(fmt, ids[i], f, idkey, r)
            qs.append((p, v if fmt in _NOSPACE else " " + v, s))
            sk.append(f)
        return _plant(w, r, facts, qs, "/".join(sk))
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 7)))
    return _finish(w, out, m, sk, want_mask)

# ================================================================ 3. constants in code
_ID_A = "MAX MIN DEFAULT BASE INITIAL FALLBACK TARGET LOCAL REMOTE PRIMARY SPARE LEGACY".split()
_ID_B = ("RETRY TIMEOUT PORT BUFFER WORKER CACHE BATCH QUEUE SHARD REPLICA THREAD POOL "
         "WINDOW CHUNK LEASE QUOTA BACKOFF SEED").split()
_ID_C = ["", "_MS", "_SIZE", "_COUNT", "_LIMIT", "_ID", "_NAME", "_KEY", "_LEVEL"]

def _ident(r):
    return "%s_%s%s" % (F2._pick(r, _ID_A), F2._pick(r, _ID_B), F2._pick(r, _ID_C))

def _camel(s):
    p = s.lower().split("_")
    return p[0] + "".join(x.capitalize() for x in p[1:])

def _cval(r):
    k = r.random()
    if k < 0.45: return F2._num(r, 2, 99999), False
    if k < 0.65: return "%d.%d" % (int(r.integers(0, 99)), int(r.integers(1, 99))), False
    if k < 0.85: return F2._phrase(r).replace(" ", "-"), True
    return F2._code(r), True

_LANGS = ["py", "js", "c", "go", "env", "rs"]
def _decl(lang, name, v, is_str):
    q = '"%s"' % v if is_str else v
    if lang == "py":  return "%s = %s" % (name, q)
    if lang == "js":  return "const %s = %s;" % (_camel(name), q)
    if lang == "c":   return "#define %s %s" % (name, q)
    if lang == "go":  return "const %s = %s" % (_camel(name)[0].upper() + _camel(name)[1:], q)
    if lang == "env": return "%s=%s" % (name, v)
    if lang == "rs":  return "const %s: %s = %s;" % (name, "&str" if is_str else ("f64" if "." in v else "u32"), q)
    raise ValueError(lang)

def _cquery(lang, name, v, is_str):
    """(prefix, answer, closer) for reading constant `name` back in that language.  The prefix never
    ends in a space: the space, where the language has one, belongs to the answer token (" 4096"),
    exactly as it does in the declaration."""
    cn = _camel(name); gn = cn[0].upper() + cn[1:]
    if lang == "py":
        return ("\n>>> %s\n'" % name, v, "'") if is_str else ("\n>>> %s\n" % name, v, "")
    if lang == "js":
        return ('\nconsole.log(%s); // prints "' % cn, v, '"') if is_str else ("\nconsole.log(%s); // prints" % cn, " " + v, "")
    if lang == "c":
        return ('\n/* %s expands to */ "' % name, v, '"') if is_str else ("\n/* %s expands to */" % name, " " + v, "")
    if lang == "go":
        return ('\nfmt.Println(%s) // "' % gn, v, '"') if is_str else ("\nfmt.Println(%s) //" % gn, " " + v, "")
    if lang == "env":
        return ("\n$ echo $%s\n" % name, v, "")
    if lang == "rs":
        return ('\nassert_eq!(%s, "' % name, v, '");') if is_str else ("\nassert_eq!(%s," % name, " " + v, ");")
    raise ValueError(lang)

_CODE_NOISE = {
    "py": ["import os", "import json", "def main():", "    pass", "# TODO: tidy this up", "logger = None", "items = []"],
    "js": ["'use strict';", "let items = [];", "function init() {}", "// helpers below", "module.exports = {};"],
    "c":  ["#include <stdio.h>", "#include <stdlib.h>", "/* config */", "static int ready = 0;", "#pragma once"],
    "go": ["package main", "import \"fmt\"", "// config values", "var ready bool", "type Cfg struct{}"],
    "env": ["# local overrides", "DEBUG=0", "LOG_FORMAT=json", "# do not commit"],
    "rs": ["use std::env;", "// constants", "#[allow(dead_code)]", "mod util;"],
}

def code_recall(w, T, rng, cue=None, want_mask=False):
    lang = _LANGS[int(rng.integers(0, len(_LANGS)))]
    def build(k):
        r = rng
        names = []
        while len(names) < k:
            x = _ident(r)
            if x not in names: names.append(x)
        vals = [_cval(r) for _ in range(k)]
        lines = []
        for nm, (v, s) in zip(names, vals):
            if r.random() < 0.35: lines.append(F2._pick(r, _CODE_NOISE[lang]))
            lines.append(_decl(lang, nm, v, s))
        facts = "\n" + "\n".join(lines) + "\n"
        qs, sk = [], ["code:%s" % lang]
        for i in _ask(r, k):
            v, s = vals[i]
            qs.append(_cquery(lang, names[i], v, s))
            sk.append(names[i])
        return _plant(w, r, facts, qs, "/".join(sk))
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 7)))
    return _finish(w, out, m, sk, want_mask)

# ================================================================ 4. schedules
_ACTS = ("meet the auditors|run the safety drill|collect the samples|sign the lease|repaint the hall|"
         "tune the piano|inspect the boiler|pitch the new design|interview the candidate|move the servers|"
         "teach the workshop|calibrate the scales|visit the depot|record the podcast|test the alarms|"
         "plant the hedges|review the budget|ship the prototype|clean the archive|host the quiz").split("|")
_VENUES = ("north hall|room 4B|the boathouse|the old library|studio two|the rooftop|the canteen|lab 7|"
           "the east gate|the chapel|bay 12|the greenhouse|the annex office|the loading dock").split("|")
_SCHED_STMT = [  # {p} person {a} activity {v} venue {d} day {t} time
    "{p} will {a} in {v} on {d} at {t}.", "On {d} at {t}, {p} will {a} in {v}.",
    "At {t} on {d} {p} is due to {a} in {v}.", "{p} is booked to {a} on {d}, {t}, in {v}.",
    "{d} {t}: {p} to {a} ({v}).",
]
_SCHED_Q = [  # (question, which slot answers it, lead-in)
    ("Q: Where will {p} {a}? A:", "v", " in"), ("Q: On which day will {p} {a}? A:", "d", " on"),
    ("Q: At what time will {p} {a}? A:", "t", " at"), ("Q: Who will {a}? A:", "p", ""),
    ("{p} will {a} on", "d", ""), ("{p} will {a} in", "v", ""),
]

def schedule_recall(w, T, rng, cue=None, want_mask=False):
    def build(k):
        r = rng
        acts = [_ACTS[int(i)] for i in r.permutation(len(_ACTS))[:k]]
        rows = [dict(p=_person(r), a=a, v=F2._pick(r, _VENUES), d=VALUES["day"](r), t=VALUES["time"](r)) for a in acts]
        stmts, sk = [], []
        for row in rows:
            si = int(r.integers(0, len(_SCHED_STMT)))
            stmts.append(_SCHED_STMT[si].format(**row)); sk.append("s%d" % si)
        facts = "\n" + " ".join(stmts) + "\n"
        qs = []
        for i in _ask(r, k):
            qi = int(r.integers(0, len(_SCHED_Q)))
            q, slot, lead = _SCHED_Q[qi]
            pre = "\n" + q.format(**rows[i]) + lead
            qs.append((pre, " " + rows[i][slot], "."))
            sk.append("q%d:%s" % (qi, rows[i]["a"]))
        return _plant(w, r, facts, qs, "sched:" + "/".join(sk))
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 6)))
    return _finish(w, out, m, sk, want_mask)

# ================================================================ 5. inventories with counts
_ITEMS = ("bolts hinges washers fuses candles batteries ropes jars lenses gaskets pulleys springs "
          "brushes ledgers spools tiles valves badges whistles chisels clamps filters magnets nozzles "
          "rivets sprockets tags tubes wicks zippers").split()
_CONTAINERS = ["crate", "box", "bin", "drawer", "tray", "case", "locker", "pallet", "sack", "cabinet"]
_INV_STMT = ["{C} {id} holds {list}.", "In {c} {id} there are {list}.", "{C} {id}: {list}.",
             "We counted {list} in {c} {id}.", "{C} {id} was packed with {list}."]
_INV_Q = [("Q: How many {it} are in {c} {id}? A:", ""), ("{C} {id} holds", "first"),
          ("Count of {it} in {c} {id}:", ""), ("Q: What is the number of {it} in {c} {id}? A:", "")]

def inventory_recall(w, T, rng, cue=None, want_mask=False):
    c = F2._pick(rng, _CONTAINERS)
    def build(k):
        r = rng
        ids = [str(x) for x in r.permutation(np.arange(1, 99))[:k]]
        boxes = []
        for _ in range(k):
            its = [_ITEMS[int(i)] for i in r.permutation(len(_ITEMS))[:int(r.integers(2, 5))]]
            boxes.append([(it, F2._num(r, 2, 400)) for it in its])
        stmts, sk = [], ["inv:%s" % c]
        for bid, b in zip(ids, boxes):
            lst = ", ".join("%s %s" % (n, it) for it, n in b[:-1]) + " and %s %s" % (b[-1][1], b[-1][0])
            si = int(r.integers(0, len(_INV_STMT)))
            stmts.append(_INV_STMT[si].format(C=c.capitalize(), c=c, id=bid, list=lst)); sk.append("s%d" % si)
        facts = "\n" + " ".join(stmts) + "\n"
        qs = []
        for i in _ask(r, k):
            qi = int(r.integers(0, len(_INV_Q)))
            q, mode = _INV_Q[qi]
            if mode == "first":          # mid-sentence: the first count is the continuation
                it, n = boxes[i][0]
                qs.append(("\n" + q.format(C=c.capitalize(), c=c, id=ids[i]), " " + n, " " + it + "."))
            else:
                it, n = boxes[i][int(r.integers(0, len(boxes[i])))]
                qs.append(("\n" + q.format(C=c.capitalize(), c=c, id=ids[i], it=it), " " + n, "."))
            sk.append("q%d:%s" % (qi, it))
        return _plant(w, r, facts, qs, "/".join(sk))
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 6)))
    return _finish(w, out, m, sk, want_mask)

# ================================================================ 6. glossary of invented words
_G_A = "zor bel quin tav mor ush pli dex ran vol gim sef".split()
_G_B = "ith ask umb ore ell yx ant iff osh erl".split()
_G_C = ["", "e", "a", "et", "o"]
_GLOSS_KIND = "tool|dance|tax|knot|bread|weather|game|ritual|fabric|rank|fish|lamp|song|vessel|coin".split("|")
_GLOSS_USE = ("sealing jars|marking graves|weighing salt|calling cattle|cutting reeds|counting days|"
              "lighting kilns|carrying water|binding books|warding off rain|trading wool|tuning bells|"
              "measuring rope|drying fish|signalling ships").split("|")
_GLOSS_STMT = ["{W} (n.): a {k} used for {u}.", "A {w} is a {k} used for {u}.",
               "{W} -- {k}; used for {u}.", "The word {w} means a {k} used for {u}."]
_GLOSS_Q = [("Q: What is a {w} used for? A: for", "u"), ("Q: What kind of thing is a {w}? A: a", "k"),
            ("A {w} is a", "k"), ("{W} (n.): a {k} used for", "u")]

def glossary_recall(w, T, rng, cue=None, want_mask=False):
    def build(k):
        r = rng
        words = []
        while len(words) < k:
            x = F2._pick(r, _G_A) + F2._pick(r, _G_B) + F2._pick(r, _G_C)
            if x not in words: words.append(x)
        rows = [dict(w=x, W=x.capitalize(), k=F2._pick(r, _GLOSS_KIND), u=F2._pick(r, _GLOSS_USE)) for x in words]
        stmts, sk = [], []
        for row in rows:
            si = int(r.integers(0, len(_GLOSS_STMT)))
            stmts.append(_GLOSS_STMT[si].format(**row)); sk.append("s%d" % si)
        facts = "\n" + " ".join(stmts) + "\n"
        qs = []
        for i in _ask(r, k):
            qi = int(r.integers(0, len(_GLOSS_Q)))
            q, slot = _GLOSS_Q[qi]
            qs.append(("\n" + q.format(**rows[i]), " " + rows[i][slot], "."))
            sk.append("q%d:%s" % (qi, rows[i]["w"]))
        return _plant(w, r, facts, qs, "gloss:" + "/".join(sk))
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 6)))
    return _finish(w, out, m, sk, want_mask)

# ================================================================ 7. object moves (latest location wins)
_MOVE_OBJ = ("toolbox|ledger|kettle|blue folder|spare tyre|ladder|first-aid kit|projector|crate of apples|"
             "violin case|road map|laptop bag|fire blanket|step stool|camera tripod|seed tray").split("|")
_MOVE_PLACE = ("the garage|the attic|the van|the porch|the cellar|the office|the shed|the hallway|"
               "the kitchen|the workshop|the car boot|the greenhouse|the stockroom|the landing").split("|")
_MOVE_STMT = ["{p} moved the {o} to {l}.", "{p} put the {o} in {l}.", "The {o} was taken to {l} by {p}.",
              "{p} carried the {o} into {l}.", "Later, {p} shifted the {o} to {l}."]
_MOVE_Q = [("Q: Where is the {o} now? A: in", ""), ("The {o} is now in", ""), ("Q: Where did the {o} end up? A: in", "")]

def moves_recall(w, T, rng, cue=None, want_mask=False):
    def build(k):
        r = rng
        objs = [_MOVE_OBJ[int(i)] for i in r.permutation(len(_MOVE_OBJ))[:k]]
        events, final, sk = [], {}, []
        for o in objs:
            for _ in range(int(r.integers(1, 4))):      # 1-3 moves; the LAST one is the answer
                events.append((o, F2._pick(r, _MOVE_PLACE)))
        order = r.permutation(len(events))
        # keep each object's moves in order relative to each other: stable sort by a random key per
        # event, then re-sequence each object's own list
        seq = [events[int(i)] for i in order]
        per = {o: [l for (oo, l) in events if oo == o] for o in objs}
        cnt = {o: 0 for o in objs}
        stmts = []
        for o, _ in seq:
            l = per[o][cnt[o]]; cnt[o] += 1; final[o] = l
            si = int(r.integers(0, len(_MOVE_STMT)))
            stmts.append(_MOVE_STMT[si].format(p=F2._pick(r, FIRST), o=o, l=l)); sk.append("s%d" % si)
        facts = "\n" + " ".join(_cap1(s) for s in stmts) + "\n"
        qs = []
        for i in _ask(r, k):
            o = objs[i]
            qi = int(r.integers(0, len(_MOVE_Q)))
            l = final[o]                                  # "in the attic": "the" is part of the answer
            qs.append(("\n" + _MOVE_Q[qi][0].format(o=o), " " + l, "."))
            sk.append("q%d:%s:%d" % (qi, o, len(per[o])))
        return _plant(w, r, facts, qs, "moves:" + "/".join(sk))
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 6)))
    return _finish(w, out, m, sk, want_mask)

def _cap1(s):
    return s[:1].upper() + s[1:]

# ================================================================ 8. quotations of real text
_QUOTE_STMT = ['{p} wrote: "{q}"', '{p} said, "{q}"', 'In the minutes, {p} is quoted: "{q}"',
               '"{q}" -- {p}', 'Note from {p}: "{q}"']
_QUOTE_Q = ['{p} wrote: "', '{p} said, "', 'Q: What did {p} say? A: "', 'The quote from {p} was: "']

def quote_recall(w, T, rng, cue=None, want_mask=False):
    """The quoted span is real text from the window (copied as TOKENS, so no re-tokenisation drift);
    speakers are distinct, so the speaker is the only key."""
    n = len(w)
    def build(k):
        r = rng
        L = int(r.integers(4, 10))
        if n < 4 * L + 64:
            return None, None, None
        people = []
        while len(people) < k:
            x = _person(r)
            if x not in people: people.append(x)
        spans = [w[s:s + L].copy() for s in r.choice(n - L - 1, size=k, replace=False)]
        ft, sk = list(_E("\n")), []
        for p, sp in zip(people, spans):
            si = int(r.integers(0, len(_QUOTE_STMT)))
            a, b = _QUOTE_STMT[si].split("{q}")
            ft += _E(a.format(p=p)) + [int(t) for t in sp] + _E(b.format(p=p)) + _E("\n")
            sk.append("s%d" % si)
        facts = np.array(ft, dtype=np.int64)
        qt, qm = [], []
        for i in _ask(r, k):
            qi = int(r.integers(0, len(_QUOTE_Q)))
            pre = _E("\n" + _QUOTE_Q[qi].format(p=people[i]))
            qt += pre + [int(t) for t in spans[i]] + _E('"')
            qm += [0.0] * len(pre) + [1.0] * L + [0.0]
            sk.append("q%d" % qi)
        q = np.array(qt, dtype=np.int64)
        room = n - len(facts) - len(q) - 32
        if room < 16:
            return None, None, None
        out, m = _splice(w, facts, q, _gap(r, 16, room), np.empty(0, dtype=np.int64), np.array(qm, dtype=np.float32))
        return out, m, "quote:%d:" % L + "/".join(sk)
    out, m, sk = _shrink(build, w, rng, int(rng.integers(2, 6)))
    return _finish(w, out, m, sk, want_mask)

# ---------------------------------------------------------------- common exit + registry
_LAST_SKELETON = [None]

def _finish(w, out, m, sk, want_mask):
    if out is None:
        _LAST_SKELETON[0] = None
        return _ret(w, None, want_mask)
    _LAST_SKELETON[0] = sk
    return _ret(out, m, want_mask)

GENERATORS = {
    "chat": chat_recall, "record": record_recall, "codeconst": code_recall, "schedule": schedule_recall,
    "inventory": inventory_recall, "glossary": glossary_recall, "moves": moves_recall, "quote": quote_recall,
}
for _k in GENERATORS:
    assert _k not in TL.GENERATORS, "name clash with task_lib: %s" % _k
TL.GENERATORS.update(GENERATORS)
