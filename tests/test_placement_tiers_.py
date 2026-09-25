"""
CPU-only tests of the placement's storage rules and new layer words (exllamav3/model/placement.py,
placement_storage.py; doc/placement.md, doc/expert_tiers.md): every worked example parses and
prints its canonical form, which parses back equal, as do its dict and JSON forms and a file
holding it; every documented parse-time refusal is the one raised; the older spellings keep their
meaning; hot= maps onto the split machinery; and the words this build cannot run yet are refused
after every other check, with their own message.

    python tests/test_placement_tiers_.py

placement.py is torch-free and loaded by path (it loads placement_storage.py and
util/host_budget.py the same way). Words pending in placement_storage.PENDING are exercised with
the gate lifted (patch.dict(PENDING, clear = True)), as the commit that makes them runnable will
do for good.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_placement_tiers_subject", ROOT / "exllamav3/model/placement.py")
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)
PS = P.storage


def lifted():
    """Parse as if every pending word were runnable"""
    return patch.dict(PS.PENDING, {}, clear = True)


# (id, meaning, as written, canonical, runnable in this build)
EXAMPLES = [
    ("T1", "one GPU, every expert in VRAM", "*=cuda:0", "*=cuda:0", True),
    ("T2", "one GPU, experts of the last ten of 60 layers on the CPU worker",
     "*=cuda:0; 50-59=cuda:0 experts=cpu", "50-59=cuda:0 experts=cpu; *=cuda:0", True),
    ("T3", "one GPU, 32 of 128 experts per layer resident, the rest on the CPU worker (-mcs 96)",
     "*=cuda:0 experts=cpu hot=32", "*=cuda:0 experts=cpu hot=32", True),
    ("T3b", "the same, as the branch writes it", "*=cuda:0 experts=split cpu=96", "*=cuda:0 experts=split cpu=96", True),
    ("T3c", "the same, in the older hybrid spelling", "*=cuda:0 experts=hybrid hot=32", "*=cuda:0 experts=cpu hot=32", True),
    ("T4", "V4.1 serving layout, CMP + PRO", "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1",
     "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1", True),
    ("T5", "V4.1, CMP + PRO, layers 12-39 cached over RAM that holds the rest, no disk reads",
     "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off",
     "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off", False),
    ("T6", "V4.1 on the PRO alone: 75 GiB cache, 96 GiB RAM tier, n-gram budget",
     "*=cuda:0 experts=cache; cuda:0 cache=75GiB; ram experts=96GiB ngram=6GiB",
     "*=cuda:0 experts=cache; cuda:0 cache=75GiB; ram experts=96GiB ngram=6GiB", False),
    ("T7", "RAM-limited box: 16 pinned per layer, the tier takes what is free after 24 GiB of page cache",
     "*=cuda:0 experts=cache hot=16; ram experts=auto pagecache=24GiB",
     "*=cuda:0 experts=cache hot=16; ram experts=auto pagecache=24GiB", False),
    ("T8", "no RAM tier: the VRAM cache straight over the SSD", "*=cuda:0 experts=cache; ram experts=0",
     "*=cuda:0 experts=cache; ram experts=0", False),
    ("T9", "per-layer overrides with profiles and prefetch",
     "0-2,39=cuda:1; 3-29=cuda:1 experts=cache hot=10% profile=code:3,wiki:1; "
     "30-38=cuda:1 experts=cache prefetch=layer:2+router; ram experts=64GiB",
     "0-2,39=cuda:1; 3-29=cuda:1 experts=cache hot=10% profile=code:3,wiki:1; "
     "30-38=cuda:1 experts=cache prefetch=layer:2+router; ram experts=64GiB", False),
    ("T10", "big-RAM host: inclusive RAM, whole n-gram tables",
     "*=cuda:0 experts=cache; ram experts=all ngram=all policy=inclusive",
     "*=cuda:0 experts=cache; ram experts=all ngram=all policy=inclusive", False),
    ("T11", "every policy knob",
     "*=cuda:0 experts=cache; cuda:0 cache=40GiB spare=12 evict=lfu admit=heat; "
     "ram experts=64GiB pagecache=16GiB policy=exclusive demote=heat evict=lru; disk io=direct",
     "*=cuda:0 experts=cache; cuda:0 cache=40GiB spare=12 evict=lfu admit=heat; "
     "ram experts=64GiB pagecache=16GiB policy=exclusive demote=heat evict=lru; disk io=direct", False),
    ("T12", "no D2H at all", "*=cuda:0 experts=cache; ram experts=48GiB demote=off",
     "*=cuda:0 experts=cache; ram experts=48GiB demote=off", False),
    ("T13", "CPU-computed layers on one GPU, cached layers on the other",
     "0-9=cuda:0 experts=cpu; 10-39=cuda:1 experts=cache; cuda:1 cache=40GiB; ram experts=96GiB",
     "0-9=cuda:0 experts=cpu; 10-39=cuda:1 experts=cache; cuda:1 cache=40GiB; ram experts=96GiB", False),
    ("T14", "experts and engram tables read from other drives",
     '*=cuda:0 experts=cache; ram experts=48GiB; disk experts=/nvme1/v41 ngram="/mnt/engram disk/v41"',
     '*=cuda:0 experts=cache; ram experts=48GiB; disk experts=/nvme1/v41 ngram="/mnt/engram disk/v41"', False),
    ("T15", "multi-line value with comments; written-out defaults vanish", """
     0-11  = cuda:0                                  # CMP: resident
     12-39 = cuda:1 experts=cache                    # PRO: cached
     cuda:1 cache=auto spare=8 evict=lru admit=adaptive
     ram experts=52GiB pagecache=auto policy=lazy-exclusive demote=swap evict=lfu
     disk experts=model ngram=model io=auto
     """, "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=52GiB", False),
    ("T16", "GPU streaming with a resident slice (-mcs k -mcm stream_only)",
     "*=cuda:1 experts=stream hot=64", "*=cuda:1 experts=stream hot=64", True),
    ("T17", "units", "*=cuda:0 experts=cache; cuda:0 cache=1.5; ram experts=48GB ngram=512mi",
     "*=cuda:0 experts=cache; cuda:0 cache=1536MiB; ram experts=48GB ngram=512MiB", False),
    ("T18", "n-gram budget only, for a PLE model", "*=cuda:0; ram ngram=12GiB", "*=cuda:0; ram ngram=12GiB", True),
    # the three test configurations of doc/expert_tiers.md
    ("a", "PRO + RAM for every non-VRAM expert", "*=cuda:0 experts=cache; ram experts=all; disk experts=off",
     "*=cuda:0 experts=cache; ram experts=all; disk experts=off", False),
    ("b", "PRO + RAM <= 128 GiB + SSD tier", "*=cuda:0 experts=cache; ram experts=96GiB",
     "*=cuda:0 experts=cache; ram experts=96GiB", False),
    ("c", "PRO + CMP resident + RAM, no expert SSD reads",
     "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off",
     "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=all; disk experts=off", False),
    # runnable storage words
    ("R1", "a request on stream layers, page cache kept for auto", "0-11=cuda:0; 12-22=cuda:1 experts=stream; "
     "23-39=cuda:1; ram experts=64GiB pagecache=16GiB",
     "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1; ram experts=64GiB pagecache=16GiB", True),
    ("R2", "whole n-gram tables, never read from disk", "*=cuda:0; ram ngram=all; disk ngram=off",
     "*=cuda:0; ram ngram=all; disk ngram=off", True),
    ("R3", "auto n-gram budget", "*=cuda:0 experts=cpu; ram experts=all ngram=auto",
     "*=cuda:0 experts=cpu; ram experts=all ngram=auto", True),
]

# (as written, full message) for parse-time refusals, with the gate lifted
PARSE_ERRORS = [
    ("rams experts=48GiB", "placement rule 1 'rams experts=48GiB': unknown rule 'rams' (did you mean 'ram'?) (a rule starts with layers (7, 0-11, 0-3,8-11), '*', 'embed', 'head', 'cuda:<n>', 'ram' or 'disk')"),
    ("*=cuda:0 experts=cache; cuda:0=75GiB", "placement rule 2 'cuda:0=75GiB': cuda:0 takes attributes, not '=<value>': write e.g. 'cuda:0 cache=75GiB'"),
    ("*=cuda:0 experts=cache; ram=48GiB", "placement rule 2 'ram=48GiB': ram takes attributes, not '=<value>': write e.g. 'ram experts=48GiB'"),
    ("*=cuda:0 experts=cache cache=75GiB", "placement rule 1 '*=cuda:0 experts=cache cache=75GiB': unknown attribute 'cache' for a layer rule (expected experts, hot, cpu, prefetch, profile); cache= belongs in a cuda:<n> rule"),
    ("*=cuda:0 experts=cache hott=16", "placement rule 1 '*=cuda:0 experts=cache hott=16': unknown attribute 'hott' for a layer rule (expected experts, hot, cpu, prefetch, profile) (did you mean 'hot'?)"),
    ("*=cuda:0 experts=cache; ram expert=48GiB", "placement rule 2 'ram expert=48GiB': unknown attribute 'expert' for the ram rule (expected experts, ngram, pagecache, policy, demote, evict) (did you mean 'experts'?)"),
    ("*=cuda:0 experts=cache; ram experts=48 GiB", "placement rule 2 'ram experts=48 GiB': 'GiB' must be key=value (experts, ngram, pagecache, policy, demote, evict) (sizes are written without spaces: 48GiB)"),
    ("*=cuda:0 experts=cache; ram experts=48G", "placement rule 2 'ram experts=48G': experts=48G: '48G' is ambiguous: write 48GiB (2^30 bytes) or 48GB (10^9 bytes)"),
    ("*=cuda:0 experts=cache; ram experts=48GIGS", "placement rule 2 'ram experts=48GIGS': experts=48GIGS: not a size (use 0, auto, all, or a number with B, KiB, MiB, GiB, TiB, KB, MB, GB or TB (a bare number is GiB))"),
    ("*=cuda:0 experts=tiered", "placement rule 1 '*=cuda:0 experts=tiered': experts='tiered' must be one of vram, cache, stream, cpu (or the older split cpu=<k> / hybrid hot=<k>)"),
    ("*=cuda:0 experts=cahce", "placement rule 1 '*=cuda:0 experts=cahce': experts='cahce' must be one of vram, cache, stream, cpu (or the older split cpu=<k> / hybrid hot=<k>) (did you mean 'cache'?)"),
    ("*=cuda:0 hot=16", "placement rule 1 '*=cuda:0 hot=16': hot= applies to experts=cache, stream and cpu (experts=vram keeps every routed expert in VRAM)"),
    ("*=cuda:0 experts=split cpu=96 hot=32", "placement rule 1 '*=cuda:0 experts=split cpu=96 hot=32': hot= applies to experts=cache, stream and cpu (experts=split counts the other side: cpu=<k>)"),
    ("*=cuda:0 experts=hybrid", "placement rule 1 '*=cuda:0 experts=hybrid': experts=hybrid needs hot=<routed experts per layer kept in VRAM> (the same as experts=cpu hot=<k>)"),
    ("*=cuda:0 experts=cpu cpu=96", "placement rule 1 '*=cuda:0 experts=cpu cpu=96': cpu= only applies to experts=split (write experts=cpu hot=<k> to keep k routed experts per layer in VRAM)"),
    ("*=cuda:0 experts=cache hot=100%", "placement rule 1 '*=cuda:0 experts=cache hot=100%': hot=100%: a share must lie between 0% and 100%, exclusive"),
    ("*=cuda:0 experts=cpu prefetch=layer", "placement rule 1 '*=cuda:0 experts=cpu prefetch=layer': prefetch= applies to experts=cache and stream (experts=cpu computes on the CPU worker)"),
    ("*=cuda:0 experts=cache prefetch=layer+layer:2", "placement rule 1 '*=cuda:0 experts=cache prefetch=layer+layer:2': prefetch lists layer twice"),
    ("*=cuda:0 experts=cache prefetch=next", "placement rule 1 '*=cuda:0 experts=cache prefetch=next': prefetch=next must be off, auto, or methods joined by '+' (layer[:<depth>], router[:<depth>])"),
    ("*=cuda:0 profile=code", "placement rule 1 '*=cuda:0 profile=code': profile= chooses hot or cached experts; experts=vram has none to choose"),
    ("*=cuda:0 experts=cache; cuda:0 cache=all", "placement rule 2 'cuda:0 cache=all': cache=all would hold every expert of its layers: write experts=vram on them"),
    ("*=cuda:0 experts=cache; cuda:0 cache=0 spare=4", "placement rule 2 'cuda:0 cache=0 spare=4': cache=0 has no slots for spare= / evict= / admit= to govern"),
    ("*=cuda:0 experts=cache; cuda:0 admit=decode", "placement rule 2 'cuda:0 admit=decode': admit=decode must be one of adaptive, heat, always"),
    ("*=cuda:0 experts=cache; cuda:1 cache=8GiB", "placement: 'cuda:1 cache=8GiB' configures an expert cache, but no experts=cache layer is placed on cuda:1"),
    ("*=cuda:0 experts=cache; cuda:0 cache=8GiB; cuda:0 spare=4", "placement rule 3 'cuda:0 spare=4': cuda:0 is set twice (rules 2 and 3); merge them into one rule"),
    ("*=cuda:0; ram experts=48GiB", "placement: 'ram experts=48GiB' but no layer keeps routed experts in system RAM (every rule is experts=vram)"),
    ("*=cuda:0 experts=cpu; ram experts=64GiB policy=exclusive", "placement: ram policy= governs the RAM tier of experts=cache layers, and there are none"),
    ("*=cuda:0 experts=cache; ram experts=0 policy=inclusive", "placement rule 2 'ram experts=0 policy=inclusive': policy= governs the RAM tier, which experts=0 disables"),
    ("*=cuda:0 experts=cache; ram experts=auto ngram=auto", "placement rule 2 'ram experts=auto ngram=auto': experts=auto and ngram=auto: at most one RAM budget can be auto"),
    ("*=cuda:0 experts=cache; ram experts=48GiB policy=lazy", "placement rule 2 'ram experts=48GiB policy=lazy': policy=lazy must be one of lazy-exclusive, exclusive, inclusive"),
    ("*=cuda:0 experts=cache; ram experts=48GiB demote=never", "placement rule 2 'ram experts=48GiB demote=never': demote=never must be one of swap, heat, all, off"),
    ("*=cuda:0 experts=cache; ram experts=48GiB policy=inclusive demote=swap", "placement rule 2 'ram experts=48GiB policy=inclusive demote=swap': demote= has no effect with policy=inclusive (every VRAM-cached expert keeps its RAM copy, so a victim is simply dropped); remove it"),
    ("*=cuda:0 experts=cache; ram experts=auto pagecache=all", "placement rule 2 'ram experts=auto pagecache=all': pagecache=all: all is not allowed here (use auto or a size)"),
    ("*=cuda:0 experts=cpu; disk experts=/nvme1/v41", "placement: 'disk experts=/nvme1/v41' applies to experts=cache layers (stream and cpu layers keep every expert in RAM), and there are none"),
    ("*=cuda:0; disk experts=off ngram=off io=direct", "placement rule 2 'disk experts=off ngram=off io=direct': io= configures disk reads, which experts=off and ngram=off both disable"),
    ("*=cuda:0; ram ngram=8GiB; disk ngram=off", "placement: 'disk ngram=off' never reads n-gram rows from disk, which needs ngram=all, not ngram=8GiB"),
    ("*=cuda:0 experts=cache; ram experts=all demote=off; disk experts=off", "placement: 'disk experts=off' keeps no copy of an expert outside VRAM and RAM, so demote=off would lose VRAM victims; drop demote=off (with the disk off, every victim moves back to the RAM slot its replacement left) or use policy=inclusive with ram experts=all"),
    ("0-19=cuda:0 experts=cpu; 20-39=cuda:0 experts=cache", "placement: cuda:0 holds both GPU-computed (experts=stream / cache) and CPU-computed (experts=cpu / split) layers; one kind of host-held experts per GPU is supported"),
    ("ram ngram=6GiB", "placement: storage rules (cuda:<n>, ram, disk) need layer rules to go with; without a placement, set RAM budgets with --expert_ram / --ngram_ram"),
    ('*=cuda:0 experts=cache profile="code', "placement: unterminated quote in '*=cuda:0 experts=cache profile=\"code'"),
    # beyond DESIGN 3.10
    ("*=cuda:0 experts=cache; cuda:0 spare=0", "placement: cuda:0 spare=0 leaves no free slot for a demoted victim; spare=0 needs ram demote=off or policy=inclusive"),
    ("*=cuda:0; disk ngram=off", "placement: 'disk ngram=off' never reads n-gram rows from disk, which needs ngram=all, not ngram=0 (the default)"),
    ("*=cuda:0 experts=stream hot=٣", "placement rule 1 '*=cuda:0 experts=stream hot=٣': hot= must be a non-negative integer"),
    ("*=cuda:0 experts=stream hot=1e1%", "placement rule 1 '*=cuda:0 experts=stream hot=1e1%': hot=1e1%: a share must lie between 0% and 100%, exclusive"),
    ("*=cuda:0 experts=cache; cuda:0 spare=-1", "placement rule 2 'cuda:0 spare=-1': spare= must be a non-negative integer"),
    ("*=cuda:0 experts=cache; cuda:0", "placement rule 2 'cuda:0': an empty cuda:0 rule; give cache=, spare=, evict= or admit="),
    ("*=cuda:0; ram", "placement rule 2 'ram': an empty ram rule; give experts=, ngram=, ..."),
    ("*=cuda:0; ram ngram=4GiB; ram experts=0", "placement rule 3 'ram experts=0': ram is set twice (rules 2 and 3); merge them into one rule"),
    ("*=cuda:0 experts=cpu; ram experts=", "placement rule 2 'ram experts=': experts= has no value"),
    ("*=cuda:0 experts=cpu; disk expert=off", "placement rule 2 'disk expert=off': unknown attribute 'expert' for the disk rule (expected experts, ngram, io) (did you mean 'experts'?)"),
    ("*=cuda:0 experts=cpu; ram cache=4GiB", "placement rule 2 'ram cache=4GiB': unknown attribute 'cache' for the ram rule (expected experts, ngram, pagecache, policy, demote, evict); cache= belongs in a cuda:<n> rule"),
    ("*=cuda:0 experts=cache; cuda:0 policy=exclusive", "placement rule 2 'cuda:0 policy=exclusive': unknown attribute 'policy' for a cuda:<n> rule (expected cache, spare, evict, admit); policy= belongs in the ram rule"),
    ("*=cuda:0; disk io=raw", "placement rule 2 'disk io=raw': io=raw must be one of auto, direct, buffered"),
]


class ExampleTests(unittest.TestCase):

    def test_canonical_and_round_trips(self):
        for eid, _, text, canon, _ in EXAMPLES:
            with self.subTest(eid), lifted():
                p = P.parse(text)
                self.assertEqual(str(p), canon)
                self.assertEqual(P.parse(str(p)), p)
                self.assertEqual(str(P.parse(str(p))), canon)
                self.assertEqual(P.parse(p.to_dict()), p)
                self.assertEqual(P.parse(json.dumps(p.to_dict())), p)
                self.assertEqual(json.loads(json.dumps(p.to_dict())), p.to_dict())

    def test_runnable_in_this_build(self):
        for eid, _, text, canon, runnable in EXAMPLES:
            with self.subTest(eid):
                if runnable:
                    self.assertEqual(str(P.parse(text)), canon)
                else:
                    with self.assertRaisesRegex(ValueError, "is not available in this build yet"):
                        P.parse(text)

    def test_equal_however_written(self):
        same = [
            "*=cuda:0 experts=cache; ram experts=48GiB policy=exclusive",
            "*=CUDA:0 Experts=Cache\nRAM Experts=48gib Policy=Exclusive  # comment",
            "*=cuda:0 experts=cache; ram experts=49152MiB policy=exclusive demote=swap evict=lfu pagecache=auto",
            "*=cuda:0 experts=cache; cuda:0 spare=8; ram experts=48Gi policy=exclusive; disk io=auto",
            '*=cuda:0 experts="cache"; ram experts=48 policy=exclusive; disk experts=model',
        ]
        with lifted():
            ps = [P.parse(s) for s in same]
        for p in ps[1:]:
            self.assertEqual(p, ps[0])
            self.assertEqual(hash(p), hash(ps[0]))
        self.assertEqual(str(ps[0]), "*=cuda:0 experts=cache; ram experts=48GiB policy=exclusive")
        # default-only storage rules vanish: the placement equals the one without them
        self.assertEqual(P.parse("*=cuda:0 experts=cpu; ram pagecache=auto; disk experts=model"),
                         P.parse("*=cuda:0 experts=cpu"))
        self.assertIsNone(P.parse("*=cuda:0 experts=cpu; ram pagecache=auto").ram)

    def test_accessors(self):
        with lifted():
            p = P.parse("0-11=cuda:0; 12-39=cuda:1 experts=cache hot=8; cuda:1 spare=12 admit=heat; "
                        "ram experts=96GiB demote=all")
        self.assertEqual(p.cache_devices(), ["cuda:1"])
        self.assertEqual((p.device_store("cuda:1").spare, p.device_store("cuda:1").admit), (12, "heat"))
        self.assertEqual((p.device_store("cuda:0").spare, p.device_store("cuda:0").cache), (8, PS.AUTO))
        self.assertEqual((str(p.ram_store().experts), p.ram_store().demote, p.ram_store().policy),
                         ("96GiB", "all", "lazy-exclusive"))
        self.assertEqual(p.disk_store(), PS.DiskStore())
        self.assertEqual(P.parse("*=cuda:0").ram_store(), PS.RamStore())
        self.assertEqual(p.experts_for_layer(20).hot, P.Hot(count = 8))

    def test_file(self):
        text = dict((e[0], e[2]) for e in EXAMPLES)["T15"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v41.placement")
            with open(path, "w") as f:
                f.write(text)
            with lifted():
                self.assertEqual(str(P.parse("@" + path)), "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=52GiB")
                self.assertEqual(str(P.parse(" @ " + path + " ")), "0-11=cuda:0; 12-39=cuda:1 experts=cache; ram experts=52GiB")
            # a file holding the JSON form, quotes and '#' inside values kept
            jpath = os.path.join(d, "p.json")
            src = {"rules": [{"layers": "*", "device": "cuda:0", "experts": "cpu"}],
                   "ram": {"experts": "all", "ngram": "6GiB"}}
            with open(jpath, "w") as f:
                json.dump(src, f)
            self.assertEqual(str(P.parse("@" + jpath)), "*=cuda:0 experts=cpu; ram experts=all ngram=6GiB")
            with lifted():
                q = P.parse('*=cuda:0 experts=cache profile="a b;#\\"c"; disk experts="/x y/#z"')
            self.assertEqual(q.experts_for_layer(0).profile, 'a b;#"c')
            self.assertEqual(q.disk.experts, "/x y/#z")
            self.assertEqual(str(q), '*=cuda:0 experts=cache profile="a b;#\\"c"; disk experts="/x y/#z"')
            with lifted():
                self.assertEqual(P.parse(str(q)), q)
        with self.assertRaisesRegex(ValueError, r"^placement: cannot read '/nonexistent/x' \(No such file or directory\)$"):
            P.parse("@/nonexistent/x")

    def test_dict_and_json_refusals(self):
        with self.assertRaisesRegex(ValueError, r"^placement: a dict / JSON placement needs a 'rules' list$"):
            P.parse({})
        with self.assertRaisesRegex(ValueError, r"^placement: unknown keys \['dsik'\] in a dict / JSON placement"):
            P.parse({"rules": [{"layers": "*", "device": "cuda:0"}], "dsik": {}})
        with self.assertRaisesRegex(ValueError, r"^placement: dict rule 1 needs 'layers' and 'device'"):
            P.parse({"rules": [{"device": "cuda:0"}]})
        with self.assertRaisesRegex(ValueError, r"^placement: invalid JSON \(Expecting"):
            P.parse('{"rules": [}')
        # a dict goes through the same checks as the text
        with self.assertRaisesRegex(ValueError, "no layer keeps routed experts in system RAM"):
            P.parse({"rules": [{"layers": "*", "device": "cuda:0"}], "ram": {"experts": "4GiB"}})

    def test_no_placement(self):
        for value in (None, "", "  ", "# nothing\n ;", ";;", "@" + os.devnull, {"rules": []},
                      {"rules": [], "devices": {}, "ram": {}}, '{"rules": []}'):
            self.assertIsNone(P.parse(value), value)
        # a word for "no placement" is refused in a file too, naming the @<path> value
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "none.placement")
            with open(path, "w") as f:
                f.write("none  # the autosplit\n")
            with self.assertRaises(ValueError) as cm:
                P.parse("@" + path)
            self.assertTrue(str(cm.exception).startswith(
                f"placement {'@' + path!r}: no placement is written as an empty value, not 'none': "), cm.exception)
        # storage rules alone are refused, not taken as no placement
        with self.assertRaisesRegex(ValueError, "need layer rules to go with"):
            P.parse({"rules": [], "ram": {"experts": "4GiB"}})


class RefusalTests(unittest.TestCase):

    def test_parse_time(self):
        for text, msg in PARSE_ERRORS:
            with self.subTest(text), lifted():
                with self.assertRaises(ValueError) as cm:
                    P.parse(text)
                self.assertEqual(str(cm.exception), msg)

    def test_pending_words(self):
        cases = (
            ("*=cuda:0 experts=cache", "experts=cache", "the expert tier runtime"),
            ("*=cuda:0 experts=stream prefetch=layer", "prefetch=", "prefill read-ahead"),
            ("*=cuda:0 experts=cpu profile=code", "profile=", "expert profiles"),
            ("*=cuda:0; disk ngram=/mnt/x", "disk ngram=<dir>", "n-gram tables read from a copy"),
            ("*=cuda:0; disk io=direct", "disk io=", "a forced read mode"),
        )
        for text, word, need in cases:
            with self.subTest(text), self.assertRaises(ValueError) as cm:
                P.parse(text)
            msg = str(cm.exception)
            self.assertTrue(msg.startswith(f"placement '{text}': "), msg)
            self.assertIn(f": {word} is not available in this build yet; it needs {need}", msg)
            self.assertTrue(msg.endswith("(see doc/expert_tiers.md)"), msg)
        self.assertEqual(PS.pending_words(P.Placement((P.Rule("rest", (), "cuda:0", P.Experts("cache")),))),
                         ["experts=cache"])
        # every other check comes first: a malformed pending word gets its own message
        with self.assertRaisesRegex(ValueError, "spare=0 leaves no free slot"):
            P.parse("*=cuda:0 experts=cache; cuda:0 spare=0")
        with self.assertRaisesRegex(ValueError, "prefetch=next must be off, auto"):
            P.parse("*=cuda:0 experts=stream prefetch=next")
        # the runnable storage words are not pending
        for text in ("*=cuda:0; ram ngram=all; disk ngram=off", "*=cuda:0 experts=cpu; ram experts=64GiB",
                     "*=cuda:0 experts=stream hot=10%; ram pagecache=12GiB ngram=auto"):
            P.parse(text)

    def test_branch_refusals_unchanged(self):
        # a sample of the branch's refusals (tests/test_placement_.py has them all), same texts
        for bad, why in (("0-3=cuda:0; 3-5=cuda:1", "layer 3 is already placed by rule 1"),
                         ("0-3=gpu:0", "device 'gpu:0' must be cuda:<n>"),
                         ("5-2=cuda:0", "range '5-2' runs backwards"),
                         ("0-3=cuda:0 experts=split", "experts=split needs cpu="),
                         ("0-3=cuda:0 experts=split cpu=²", "cpu= must be a positive integer"),
                         ("٣=cuda:0", "must look like 7, 0-11 or 0-3,8-11"),
                         ("head=cuda:0 experts=cpu", "experts= applies to decoder layers, not to head")):
            with self.subTest(bad), self.assertRaisesRegex(ValueError, why):
                P.parse(bad)


class PlanTests(unittest.TestCase):

    def test_branch_strings_keep_text_and_plans(self):
        for s in ("*=cuda:0; 50-59=cuda:0 experts=cpu", "*=cuda:0 experts=split cpu=96", "0-23=cuda:0; 24-47=cuda:1",
                  "0-11=cuda:0; 12-22=cuda:1 experts=stream; 23-39=cuda:1",
                  "0-11=cuda:0; 12-22=cuda:1 experts=cpu; 23-39=cuda:1", "0-19=cuda:1; 20-39=cuda:0; head=cuda:1",
                  "0-14=cuda:0; 15-29=cuda:1; 30-44=cuda:2 experts=stream; 45-59=cuda:3 experts=cpu",
                  "0-23=cuda:0; 24-47=cuda:1 experts=split cpu=64"):
            p = P.parse(s)
            self.assertEqual(P.parse(str(p)), p)
            self.assertEqual((p.devices, p.ram, p.disk), ((), None, None))
        x = P.parse("0-9=cuda:0; 10-19=cuda:1 experts=stream; 20-29=cuda:2 experts=cpu; "
                    "30-39=cuda:2 experts=split cpu=48")
        self.assertEqual([P.expert_plan(x, i, 64) for i in (0, 10, 20, 30)],
                         [("vram", None, 0), ("ram", "stream", 0), ("ram", "hybrid", 0), ("split", "hybrid", 48)])

    def test_hot(self):
        plan = lambda text, e: P.expert_plan(P.parse(text), 3, e)
        self.assertEqual(plan("*=cuda:0 experts=cpu hot=32", 128), ("split", "hybrid", 96))
        self.assertEqual(plan("*=cuda:0 experts=hybrid hot=32", 128), ("split", "hybrid", 96))
        self.assertEqual(plan("*=cuda:1 experts=stream hot=64", 384), ("split", "stream", 320))
        self.assertEqual(plan("*=cuda:1 experts=stream hot=0", 384), ("ram", "stream", 0))
        self.assertEqual(plan("*=cuda:1 experts=stream hot=25%", 64), ("split", "stream", 48))
        self.assertEqual(plan("*=cuda:1 experts=cpu hot=12.5%", 12), ("split", "hybrid", 10))     # 1.5 rounds up
        self.assertEqual(plan("*=cuda:1 experts=cpu hot=1%", 16), ("ram", "hybrid", 0))           # rounds to none
        self.assertEqual(str(P.parse("*=cuda:0 experts=cpu hot=0")), "*=cuda:0 experts=cpu")
        self.assertEqual(str(P.parse("*=cuda:0 experts=cpu hot=12.50%")), "*=cuda:0 experts=cpu hot=12.5%")
        with lifted():
            self.assertEqual(plan("*=cuda:0 experts=cache hot=25%", 64), ("cache", "gpu", 16))
            self.assertEqual(plan("*=cuda:0 experts=cache", 64), ("cache", "gpu", 0))
            with self.assertRaisesRegex(ValueError, r"^placement: layer 3 experts=cache hot=64 keeps all 64 routed "
                                                    r"experts in VRAM: use experts=vram$"):
                plan("*=cuda:0 experts=cache hot=64", 64)
        with self.assertRaisesRegex(ValueError, "keeps all 16 routed experts in VRAM"):
            plan("*=cuda:0 experts=stream hot=99%", 16)
        # hot= keeps its kind: stream hot is GPU-computed, cpu hot CPU-computed
        with self.assertRaisesRegex(ValueError, "one kind of host-held experts per GPU"):
            P.parse("0-3=cuda:0 experts=stream hot=4; 4-7=cuda:0 experts=cpu hot=4")
        P.parse("0-3=cuda:0 experts=stream hot=4; 4-7=cuda:0 experts=stream; *=cuda:1 experts=cpu hot=2")

    def test_every_mode_is_mapped(self):
        self.assertEqual(set(P._HOST_KIND), set(P.EXPERT_MODES))
        self.assertEqual(set(P._PLAN), set(P.EXPERT_MODES))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
