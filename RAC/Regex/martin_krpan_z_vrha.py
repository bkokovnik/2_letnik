# =============================================================================
# Martin Krpan z Vrha
#
# Pri tej nalogi bomo napisali nekaj funkcij, ki nam bodo v pomoč pri
# analizi literarnih besedil, kot je na primer Levstikova povest
# [Martin Krpan z Vrha](http://lit.ijs.si/martinkr.html).
# 
#     odlomek = """V Notranjem stoji vas, Vrh po imenu. V tej vasici je
#     živel v starih časih Krpan, močan in silen človek. Bil je neki
#     tolik, da ga ni kmalu takega. Dela mu ni bilo mar; ampak nosil je
#     od morja na svoji kobilici angleško sol, kar je bilo pa že tistikrat
#     ostro prepovedano. Pazili so ga mejači, da bi ga kje nehotoma zalezli;
#     poštenega boja ž njim so se bali ravno tako kakor pozneje Štempiharja.
#     Krpan se je pa vedno umikal in gledal, da mu niso mogli do živega."""
# =====================================================================@005905=
# 1. podnaloga
# Sestavite funkcijo `najdi_besede(besedilo, podniz)`, ki vrne množico
# vseh besed, ki se pojavijo v nizu `besedilo` in vsebujejo niz `podniz`.
# Zgled:
# 
#     >>> najdi_besede(odlomek, 'il')
#     {'silen', 'bilo', 'Bil', 'nosil', 'Pazili', 'kobilici'}
# =============================================================================

"""
a   znak "a"
.   Katerikoli znak
a|b   Pomeni a ali b
[abc] Pomeni katerikoli znak izmed a, b, c, lahko damo tudi A-Z, a-Z, 3-9, ... in bo našlo katerokoli iz tega razpona, za slovenščino npr a-zčšž
[^abc]  Vse razen a, b, c
a?  0 ali 1 ponovitev
a+  1 ali več ponovitev
a*  0 ali več ponovitev
a{n}    Natanko n ponovitev a
a{n,m}  Med n in vključno m ponovitev a
a{n,}   n ali več ponovitev a
ab  a in b
(abc)   Skupina
\w  Znaki besed (črke, številke, podčrtaj in čudne pismenke, čžš)
\d  Številak
\s  Whitespace (presledki, tabulatorji, ...)
\b  "Border", rob besede
Velike črke (\W, \D, \S \B) pomeni negacijo
\1  "Tu želimo vsebino prve skupine še enkrat"

"""

import re

odlomek = """V Notranjem stoji vas, Vrh po imenu. V tej vasici je živel v starih
časih Krpan, močan in silen človek. Bil je neki tolik, da ga ni kmalu takega.
Dela mu ni bilo mar; ampak nosil je od morja na svoji kobilici angleško sol, kar
je bilo pa že tistikrat ostro prepovedano. Pazili so ga mejači, da bi ga kje
nehotoma zalezli; poštenega boja ž njim so se bali ravno tako kakor pozneje
Štempiharja. Krpan se je pa vedno umikal in gledal, da mu niso mogli do živega."""


def najdi_besede(besedilo, podniz):
    vzorec = r"\b\w*" + podniz + r"\w*\b"
    return set(re.findall(vzorec, besedilo))

# =====================================================================@005906=
# 2. podnaloga
# Sestavite funkcijo `najdi_predpono(besedilo, predpona)`, ki vrne množico
# vseh besed, ki se pojavijo v nizu `besedilo` in imajo predpono `predpona`.
# Zgled:
# 
#     >>> najdi_predpono(odlomek, 'po')
#     {'pozneje', 'po', 'poštenega'}
# =============================================================================
def najdi_predpono(besedilo, predpona):
    vzorec = r"\b" + predpona + r"\w*\b"
    return set(re.findall(vzorec, besedilo))  
# =====================================================================@005907=
# 3. podnaloga
# Sestavite funkcijo `najdi_pripono(besedilo, pripona)`, ki vrne množico
# vseh besed, ki se pojavijo v nizu `besedilo` in imajo pripono `pripona`.
# Zgled:
# 
#     >>> najdi_pripono(odlomek, 'ga')
#     {'takega', 'ga', 'poštenega', 'živega'}
# =============================================================================
def najdi_pripono(besedilo, pripona):
    vzorec = r"\b\w*" + pripona + r"\b"
    return set(re.findall(vzorec, besedilo))  
# =====================================================================@005908=
# 4. podnaloga
# Sestavite funkcijo `podvojene_crke(besedilo)`, ki sprejme niz `besedilo`
# in vrne množico vseh besed, ki vsebujejo podvojene črke. Zgled:
# 
#     >>> podvojene_crke('A volunteer is worth twenty pressed men.')
#     {'pressed', 'volunteer'}
# =============================================================================
def podvojene_crke(besedilo):
    # vzorec = r"\b\w*" + r"(\w)\1" + r"\w*\b"  To bi nam vrnilo samo par črk, ker vrne samo skupino, ki smo jo definirali
    vzorec = r"(\b\w*(\w)\2\w*\b)"
    return {beseda for beseda, crka in re.findall(vzorec, besedilo)}





































































































# ============================================================================@
# fmt: off
"Če vam Python sporoča, da je v tej vrstici sintaktična napaka,"
"se napaka v resnici skriva v zadnjih vrsticah vaše kode."

"Kode od tu naprej NE SPREMINJAJTE!"

# isort: off
import json
import os
import re
import shutil
import sys
import traceback
import urllib.error
import urllib.request
import io
from contextlib import contextmanager


class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end="")
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end="")
        return line


class TimeoutError(Exception):
    pass


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part["solution"].strip() != ""

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part["valid"] = True
            part["feedback"] = []
            part["secret"] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part["feedback"].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part["valid"] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(
                Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed)
            )
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted(
                [
                    (Check.clean(k, digits, typed), Check.clean(v, digits, typed))
                    for (k, v) in x.items()
                ]
            )
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get("clean", clean)
        Check.current_part["secret"].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error(
                "Izraz {0} vrne {1!r} namesto {2!r}.",
                expression,
                actual_result,
                expected_result,
            )
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error("Namestiti morate numpy.")
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error("Ta funkcija je namenjena testiranju za tip np.ndarray.")

        if env is None:
            env = dict()
        env.update({"np": np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error(
                "Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                type(expected_result).__name__,
                type(actual_result).__name__,
            )
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error(
                "Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.",
                exp_shape,
                act_shape,
            )
            return False
        try:
            np.testing.assert_allclose(
                expected_result, actual_result, atol=tol, rtol=tol
            )
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        exec(code, global_env)
        errors = []
        for x, v in expected_state.items():
            if x not in global_env:
                errors.append(
                    "morajo nastaviti spremenljivko {0}, vendar je ne".format(x)
                )
            elif clean(global_env[x]) != clean(v):
                errors.append(
                    "nastavijo {0} na {1!r} namesto na {2!r}".format(
                        x, global_env[x], v
                    )
                )
        if errors:
            Check.error("Ukazi\n{0}\n{1}.", statements, ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, "w", encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part["feedback"][:]
        yield
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n    ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}",
                filename,
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part["feedback"][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get("stringio")("\n".join(content) + "\n")
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n  ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}",
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error(
                "Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}",
                filename,
                (line_width - 7) * " ",
                "\n  ".join(diff),
            )
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        too_many_read_requests = False
        try:
            exec(expression, global_env)
        except EOFError:
            too_many_read_requests = True
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal and not too_many_read_requests:
            return True
        else:
            if too_many_read_requests:
                Check.error("Program prevečkrat zahteva uporabnikov vnos.")
            if not equal:
                Check.error(
                    "Program izpiše{0}  namesto:\n  {1}",
                    (line_width - 13) * " ",
                    "\n  ".join(diff),
                )
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ["\n"]
        else:
            expected_lines += (actual_len - expected_len) * ["\n"]
        equal = True
        line_width = max(
            len(actual_line.rstrip())
            for actual_line in actual_lines + ["Program izpiše"]
        )
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append(
                "{0} {1} {2}".format(
                    out.ljust(line_width), "|" if out == given else "*", given
                )
            )
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get("update_env", update_env):
            global_env = dict(global_env)
        global_env.update(Check.get("env", env))
        return global_env

    @staticmethod
    def generator(
        expression,
        expected_values,
        should_stop=None,
        further_iter=None,
        clean=None,
        env=None,
        update_env=None,
    ):
        from types import GeneratorType

        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error(
                        "Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                        iteration,
                        expression,
                        actual_value,
                        expected_value,
                    )
                    return False
            for _ in range(Check.get("further_iter", further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get("should_stop", should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print("{0}. podnaloga je brez rešitve.".format(i + 1))
            elif not part["valid"]:
                print("{0}. podnaloga nima veljavne rešitve.".format(i + 1))
            else:
                print("{0}. podnaloga ima veljavno rešitev.".format(i + 1))
            for message in part["feedback"]:
                print("  - {0}".format("\n    ".join(message.splitlines())))

    settings_stack = [
        {
            "clean": clean.__func__,
            "encoding": None,
            "env": {},
            "further_iter": 0,
            "should_stop": False,
            "stringio": VisibleStringIO,
            "update_env": False,
        }
    ]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs)) if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get("env"))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get("stringio"):
            yield
        else:
            with Check.set(stringio=stringio):
                yield

    @staticmethod
    @contextmanager
    def time_limit(timeout_seconds=1):
        from signal import SIGINT, raise_signal
        from threading import Timer

        def interrupt_main():
            raise_signal(SIGINT)

        timer = Timer(timeout_seconds, interrupt_main)
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutError
        finally:
            timer.cancel()


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        part_regex = re.compile(
            r"# =+@(?P<part>\d+)=\s*\n"  # beginning of header
            r"(\s*#( [^\n]*)?\n)+?"  # description
            r"\s*# =+\s*?\n"  # end of header
            r"(?P<solution>.*?)"  # solution
            r"(?=\n\s*# =+@)",  # beginning of next part
            flags=re.DOTALL | re.MULTILINE,
        )
        parts = [
            {"part": int(match.group("part")), "solution": match.group("solution")}
            for match in part_regex.finditer(source)
        ]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]["solution"] = parts[-1]["solution"].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = "{0}.{1}".format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    "part": part["part"],
                    "solution": part["solution"],
                    "valid": part["valid"],
                    "secret": [x for (x, _) in part["secret"]],
                    "feedback": json.dumps(part["feedback"]),
                }
                if "token" in part:
                    submitted_part["token"] = part["token"]
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode("utf-8")
        headers = {"Authorization": token, "content-type": "application/json"}
        request = urllib.request.Request(url, data=data, headers=headers)
        # This is a workaround because some clients (and not macOS ones!) report
        # <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1129)>
        import ssl

        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, context=context)
        # When the issue is resolved, the following should be used
        # response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response["attempts"]:
            part["feedback"] = json.loads(part["feedback"])
            updates[part["part"]] = part
        for part in old_parts:
            valid_before = part["valid"]
            part.update(updates.get(part["part"], {}))
            valid_after = part["valid"]
            if valid_before and not valid_after:
                wrong_index = response["wrong_indices"].get(str(part["part"]))
                if wrong_index is not None:
                    hint = part["secret"][wrong_index][1]
                    if hint:
                        part["feedback"].append("Namig: {}".format(hint))

    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTA1LCJ1c2VyIjo4MDYxfQ:1s4eJr:0rPCjIm7o640_9GKnDoptRBzMgOfXZvgkdvBTuJMvZM"
        try:
            _martin_krpan_z_vrha = """Fran Levstik
            
            Martin Krpan z Vrha
            
            Močilar mi je časi kaj razkladal od nekdanjih časov, kako so ljudje živeli in kako so imeli to in to reč med sabo. Enkrat v nedeljo popoldne mi je v lipovi senci na klopi pravil naslednjo povest:
            
            V Notranjem stoji vas, Vrh po imenu. V tej vasici je živel v starih časih Krpan, močan in silen človek. Bil je neki tolik, da ga ni kmalu takega. Dela mu ni bilo mar; ampak nosil je od morja na svoji kobilici angleško sol, kar je bilo pa že tistikrat ostro prepovedano. Pazili so ga mejači, da bi ga kje nehotoma zalezli; poštenega boja ž njim so se bali ravno tako kakor pozneje Štempiharja. Krpan se je pa vedno umikal in gledal, da mu niso mogli do živega.
            
            Bilo je pozimi in sneg je ležal krog in krog. Držala je samo ozka gaz, ljudem dovoljna, od vasi do vasi, ker takrat še ni bilo tako cest kakor dandanes. V naših časih je to vse drugače, seveda; saj imamo, hvalo Bogu, cesto do vsakega zelnika. Nesel je Krpan po ozki gazi na svoji kobilici nekoliko stotov soli; kar mu naproti prižvenketa lep voz; na vozu je pa sedel cesar Janez, ki se je ravno peljal v Trst. Krpan je bil kmečki človek, zato ga tudi ni poznal; pa saj ni bilo časa dolgo ozirati se; še odkriti se ni utegnil, temveč prime brž kobilico in tovor ž njo pa jo prenese v stran, da bi je voz ne podrl. Menite, da je Krpana to kaj mudilo kali? Bilo mu je, kakor komu drugemu stol prestaviti.
            
            Cesar, to videvši, veli kočijažu, da naj konje ustavi. Ko se to zgodi, vpraša silnega moža: "Kdo pa si ti?"
            
            Ta mu dá odgovor: "Krpan mi pravijo; doma sem pa z Vrha od Svete Trojice, dve uri hoda od tukaj."
            
            "I kaj pa nosiš v tovoru?" cesar dalje vpraša.
            
            Krpan se naglo izmisli in reče: "I kaj? Kresilno gobo pa nekaj brusov sem naložil, gospod!"
            
            Na to se cesar začudi in pravi: "Ako so brusi, pokaj so pa v vrečah?"
            
            Krpan se ne umišlja dolgo, ampak urno odgovori kakor vsak človek, ki ve, kaj pravi: "Bojim se, da bi od mraza ne razpokali; zato sem jih v slamo zavil in v vrečo potisnil."
            
            Cesar, ki mu je bil menda silni možak všeč, dalje pravi: "Anti veš, kako se taki reči streže. Kaj pa, da si konjiča tako lahko prestavil? Res nima dosti mesa; pa ima vsaj kosti."
            
            Krpan se malo zareži in pravi: "Vem, da imajo vaši konji več mesa; pa vendar ne dam svoje kobilice za vse štiri, ki so tukaj napreženi. Kar se pa tiče prestavljanja, gospod, upam si nesti dve taki kobili dve uri hoda in tudi še dalj, če je treba."
            
            Cesar si misli: To velja, da bi se zapomnilo, -- in veli pognati.
            
            Minilo je potem leto in nekateri dan. Krpan je pa zmerom tovoril po hribih in dolinah. Kar se pripeti, da pride na Dunaj strašen velikan, Brdavs po imenu. Ta je vabil kakor nekdanji Pegam vse junake našega cesarstva v boj. Ali cesar pa tudi ni imel tako boječih ljudi, da bi dejal: nihče si ni upal nadenj; toda kdor se je skusil ž njim, gotovo je bil zmagan. Velikan pa ni bil mož usmiljenega srca; ampak vsakega je umoril, kogar je obvladal. -- To je cesarju začelo iti po glavi: "Lejte-si no! Kaj bo, kaj bo, če se Brdavs ne ukroti? Usmrtil mi je že vso največjo gospodo! Presneta reč vendar, da mu nihče ne more biti kos!" Tako je cesar toževal, kočijaž ga je pa slišal. Pristopi tedaj z veliko ponižnostjo, kakor gre pred tolikim gospodom, in pravi: "Cesarost, ali več ne morete pametovati, kaj se je godilo predlansko zimo blizu Trsta?"
            
            Cesar vpraša nekoliko nevoljen: "Kaj neki?"
            
            Kočijaž odgovori: "Tisti Krpan, ki je tovoril s kresilno gobo in brusi, ne veste, kako je kobilico v sneg prestavil, kakor bi nesel skledo na mizo? Če ne bo Krpan Brdavsa premogel, drugi tudi ne, tako vam povem."
            
            "Saj res," pravi cesar, "precej se mora ponj poslati."
            
            Poslali so veliko, lepo kočijo po Krpana. To je bil ravno tačas natovoril nekoliko soli pred svojo kočo: mejači so bili pa vse čisto ovédeli, da se zopet napravlja po kupčiji. Pridejo tedaj nadenj ter se ga lotijo; bilo jih je petnajst. Ali on se jih ni ustrašil; pisano je pogledal in prijel prvega in druge ž njim omlatil, da so vsi podplate pokazali. Ravno se je to vršilo, kar se v četver pripelja nova, lepa kočija. Iz nje stopi cesarski sel, ki je vse videl, kar se je godilo, in naglo reče: "Zdaj pa že vem, da sem prav pogodil. Ti si Krpan z Vrha od Svete Trojice, kajne?"
            
            "Krpan sem," pravi ta; "z Vrha tudi, od Svete Trojice tudi. Ali kaj pa bi radi? Če mislite zavoljo soli kaj, svetujem, da mirujete; petnajst jih je bilo, pa se jih nisem bal, hvalo Bogu; samo enega se tudi ne bom."
            
            Sel pa, ki gotovo ni vedel, zakaj se meni od soli, reče na to: "Le urno zapri kobilo v konják, pa se hitro pražnje obleci, pojdeva na Dunaj do cesarja."
            
            Krpan ga neverno pogleda in odgovori: "Kdor če iti na Dunaj, mora pustiti trebuh zunaj, to sem slišal od starih ljudi; jaz ga pa menim s sabo nositi, koder bom tovoril in dokler bom tovoril."
            
            Služabnik mu pravi: "Nikar ti ne misli, da šale uganjam."
            
            "Saj bi tudi ne bilo zdravo," reče Krpan.
            
            Na to zopet govori sel: "Kar sem ti povedal, vse je res. Ali več ne veš, kako si bil umaknil predlansko zimo kobilico kočiji s pota? Oni gospod na vozu je bil cesar, pa nihče drug, veš."
            
            Krpan se začudi in pravi: "Cesar? -- Menda vendar ne?"
            
            "Cesar, cesar! Le poslušaj. Prišel je zdaj na Dunaj hud velikan, ki mu pravimo Brdavs. Tak je, da ga nihče ne ustrahuje. Dosti vojščakov in gospode je že pobil; pa smo rekli: ,Če ga živ krst ne zmore, Krpan ga bo.' Lej, ti si zadnje upanje cesarjevo in dunajskega mesta."
            
            Krpana je to s pridom utešilo ter jako dobro se mu je zdelo do vsega, kar je slišal, in reče tedaj: "Če ni drugega kakor tisti prekleti Brdavs, poslušajte, kaj vam pravim! Petnajst Brdavsov za malo južino, to je meni toliko, kolikor vam kamen poriniti čez lužo, ki jo preskoči dete sedem let staro; samo varite, da me ne boste vodili za nos!" To reče in brž dene sol s kobile, kobilo pa v konják, gre v kočo ter se pražnje obleče, da bi ga pred cesarjem ne bilo sram. Ko se preobuje, ven priteče in sede v kočijo ter naglo zdrčita proti Dunaju.
            
            Ko prideta na Dunaj, bilo je vse mesto črno pregrnjeno; ljudje so pa klavrno lazili kakor mravlje, kadar se jim zapali mravljišče.
            
            Krpan vpraša: "Kaj pa vam je, da vse žaluje?"
            
            "O, Brdavs, Brdavs!" Vpije malo in veliko, možje in žene. "Ravno danes je umoril cesarjevega sina, ki ga je globoko v srce pekla sramota, da bi ne imela krona junaka pod sabo, kateri bi se ne bal velikana. Šel se je ž njim skusit; ali kaj pomaga! Kakor drugim, tako njemu. Do zdaj se še nihče ni vrnil iz boja."
            
            Krpan veli urno pognati in tako prideta na cesarski dvor, ki pavijo, da je neki silo velik in jako lep. Tam stoji straža vedno pri vratih noč in dan, v letu in zimi, naj bo še tako mraz; in brž je zavpila o Krpanovem prihodu, kakor imajo navado, kadar se pripelja kdo cesarske rodovine. Bilo je namreč naročeno že štirinajst dni dan za dnevom, da naj se nikomur in nikoli ne oglasi, samo tačas, kadar se bo pripeljal tak in tak človek. Tako so se veselili Krpana na Dunaj. Kaj bi se ga pa ne? Presneto jim je bila huda za nohti! Ko cesar sliši vpitje, precej ve, kdo je, in teče mu naproti, pa ga pelja v gornje hrame. Čudno lepo je tam, še lepše kakor v cerkvi. Krpan je kar zijal, ker se mu je vse tako grobo zdelo. Cesar ga vpraša: "Krpan z Vrha! Ali me še poznaš?"
            
            Kaj bi vas ne," odgovori on; "saj ni več ko dve leti, kar sva se videla. No vi ste še zmerom lepo zdravi, kakor se na vašem licu vidi."
            
            Cesar pravi: "Kaj pomaga ljubo zdravje, ko pa drugo vse narobe gre! Saj si že slišal od velikana? Kaj deš ti, kaj bo iz tega, če se kako kaj ne preonegavi? Sina mi je ubil, lej!"
            
            Krpan odgovori: "Koga bo drugega? Glavo mu bomo vzeli, pa je!"
            
            Cesar žalosten zavrne: "Menim da, ko bi jo le mogli! Oh, ali ni ga, mislim, pod soncem jujaka, da bi vzel Brdavsu glavo!"
            
            "Zakaj ne? Slišal sem," pravi Krpan, "da vsi ljudje vse vedo; na vsem svetu se pa vse dobi; pa bi se ne dobil tudi junak nd Brdavsa? Kakor sem uboren človek, ali tako peklensko ga bom premikastil, da se mu nikdar več ne bodo vrnile hudobne želje, po Dunaju razsajati; če Bog dá, da je res!"
            
            Kdo bi bil cesarju bolj ustregel kakor te besede! Le nekaj ga je še skrbelo; zato pa tudi reče: "Da si močan, tega si me preveril; ali pomisli ti: on je orožja vajen iz mladih dni; ti pak si prenašal zdaj le bruse in kresilno gobo po Kranjskem; sulice in meča menda še nisi videl nikoli drugje kakor na križevem potu v cerkvi. Kako se ga boš pa lotil?"
            
            "Nič se ne bojte," pravi Krpan; "kako ga bom in s čim ga bom, to je moja skrb. Ne bojim se ne meča ne sulice ne drugega velikanovega orožja, ki vsemu še imena ne vem, če ga ima kaj veliko na sebi."
            
            Vse to je bilo cesarju pogodu, in brž veli prinesti polič vina pa kruha in sira, rekoč: "Na, Krpan, pij pa jej! Potlej pojdeva orožje izbirat."
            
            Krpanu se je to vele malo zdelo; polič vina takemu junaku; pa je vendar molčal, kar je preveliko čudo. Kaj pa je hotel? Saj menda je že slišal, da gospoda so vsi malojedni zato, ker jedo, kadar hoče in kolikor hoče kateri, zgolj dobrih jedi. Ali kmečki človek, kakor je bil Krpan, ima drugo za bregom. On tedaj použije, ko bi kvišku pogledal ter naglo vstane. Cesar je vse videl in, ker je bil pameten mož, tudi precej spoznal, da takemu truplu se morajo večji deleži meriti; zato so mu pa dajali od tega časa dan na dan, dokler je bil na Dunaju: dve krači, dve četrti janjca, tri kopune, in ker sredice ni jedel, skorje štirih belih pogač, z maslom in jajci oméšanih; vino je imel pa na pravici, kolikor ga je mogel.
            
            Ko prideta v orožnico, to je v tisto shrambo, kjer imajo orožje, namreč: sabolje, meče, jeklene oklepe za na prsi, čelade in kakor se imenuje to in ono; Krpan izbira in izbira, pa kar prime, vse v rokah zdrobi, ker je bil silen človek. Cesarja skoraj obide zona, ko to vidi; vendar se stori srčnega in vpraša: "No, boš kaj kmalu izbral?"
            
            "V čem si bom pa izbiral?" odgovori Krpan. "To je sama igrača; to ni za velikana, ki se mu pravi Brdavs, pa tudi ne za mene, ki mi pravite Krpan. Kje imate kaj boljega?"
            
            Cesar se čudi in pravi: "Če to ne bo zate, sam ne vem, kako bi? Večjega in boljega nimamo."
            
            Na to reče oni: "Veste kaj? Pokažite mi, kje je katera kovačnica!"
            
            Pelja ga hitro sam cesar v kovačnico, ki je bila tudi na dvoru; zakaj taki imajo vso pripravo in tudi kovačnico, da je kladivo in nakovalo pri rokah, ako se konj izbosi ali če je kaj drugega treba, da se podstavi ali prekuje. Krpan vzame kos železa in najtežje kladivo, ki ga je kovač vselej z obema rokama vihtel; njemu je pa v eni roki pelo, kakor bi koso klepal. "Oj tat sežgani!" pravijo vsi, ko to vidijo; še cesarju se je imenitno zdelo, da ima takega hrusta pri hiši. Krpan kuje in kuje, goni meh na vse kriplje ter naredi veliko reč, ki ni bila nobenemu orožju podobna; imela je največ enakosti z mesarico. Ko to izgotovi, gre na cesarski vrt in poseka mlado, košato lipo iznad kamnite mize, kamor so hodili gospoda poleti hladit se. Cesar, ki mu je bil zmerom za petami, brž priteče in zavpije: "Krpan! I kaj pa to delaš? Da te bes opali! Ne veš, da cesarica raje dá vse konje od hiše kakor to lipo od mize? Pa si jo posekal! Kaj bo pa zdaj?"
            
            Krpan z Vrha pa, ne da bi se bal, odgovori: "Kar je, to je. Zakaj pa mi niste druge pokazali, če se bam te tako smili? Kaj bo pa? Drevo je drevo! Jaz pa moram imeti les nalašč za svojo rabo, kakršnega v boju potrebujem."
            
            Cesar molči, ker vidi, da ne pomaga zvoniti, ko je toča že pobila; pa vendar ga je skrbelo, kako bi se izgovoril pred cesarico. Krpan tedaj naredi najprvo toporišče mesarici, potem pa obseka pol sežnja dolg ter na enem koncu jako debel kij, pa gre pred cesarja: "Orožje imam, ali konja nimam. Saj menda se ne bova peš lasala?"
            
            Cesar, zastran lipe še zmerom nekoliko nevšečen, pravi: "Pojdi, pa vzemi konja, katerega hočeš. Saj vem, da le širokoustiš. Kdaj bom jaz papež v Rimu? Takrat, kadar boš ti zmogel velikana. Če misliš, primi ga, pa mu odstrizi glavo, ako si za kaj, da bo imela moja država mir pred njim, ti pa veliko čast in slavo za njim!"
            
            Krpan je bil malo srdit, pa vendar jezo pogoltne in reče: "Kar se tiče Brdavsa, to ni igrača, kakor bi kdo z grma zapodil vrabca, ki se boji vsakega ocepka in kamna. Koliko junakov pa imate, da bi si upali nádenj? Zapomnite si, cesarost, kar sem obljubil, storil bom, čeprav od jeze popokajo vsi obrekovalci, ki me mrazijo pri vas. Da bi le vsi ljudje vselej držali se svojih besedi tako, kakor se mislim jaz, ako me Bog ne udari; pa bi nihče ne vedel, kaj se pravi laž na zemlji! Toda svet je hudoben ter ne pomisli, da je Bog velik, človek majhen. Zdaj pa le pojdite, greva, da konja izbereva. Nočem takega, da bi pod mojo težo pred velikanom počenil na vse štiri noge, vam v sramoto, meni v sitnost. Dunajčanje bi se smejali, vi pa rekli: 'Poglejte ga, še konja mi je izpridil!'"
            
            Cesar je kar obstekel, poslušajo modrost Martinovih ust, in potem gre ž njim. Ko prideta v konják, povpraša: "Po čem bodeš pa konja poznal, je li dober ali ne?"
            
            Krpan odgovori: "Po tem, da se mi ne bo dal za rep čez prag potegniti."
            
            Cesar pravi: "Le skusi! Ali daravno si, prekanjeni tat, storil mi dovolj sitnosti pred cesarico, svarim te, vari se, da te kateri ne ubije; konji so iskri."
            
            Martin Krpan pak izleče prvega in zadnjega in vse druge čez prag; še celo tistega, ki ga je sam cesar jahal samo dvakrat v letu, namreč: o veliki noči pa o svetem Telesu; to se je menda cesarju posebno pod nos pokadilo. Potem reče Krpan: Tukaj ga nimate za moje sedlo! Pojdiva k drugim."
            
            Cesar odgovori čméren: "Če niso ti zate, moraš se peš bojevati. Ti nisi pravdanski človek! Vem, da ga nimam v cesarstvu takega, da bi ga ti, zagovédnež, ne izlekel!"
            
            "Ta je pa že prazna!" pravi Krpan. "Jaz imam doma kobilico, katere ne izleče nobeden vaših junakov, stavim svojo glavo, če ni drugače; da ne poreko Dunajčanje z Brdavsom vred, da lažem."
            
            "Pa ni tista," vpraša cesar, "ki si ž njo plesal po snegu?"
            
            "Tista, tista!" zavrne on.
            
            Cesar pa se razhudi, rekoč: "Zdaj pa že vidim, da si bebec ali pa mene delaš bebca! Vari se me, Krpane! Moja roka je dolga."
            
            Krpan pa mu v smehu odgovori: "Če je s tem dalja, pa vendar ne seže velikanu še celo do pasa ne, nikar že do brade, da bi ga malo oskubla in zlasala. Ampak pustimo šale takim ljudem v napotje, ki nimajo drugega dela, kakor da ž njimi dražijo svojega bližnjega; meniva se raje od Brdavsa, ki še zdaj nosi glavo. Pošljite mi hitro po kobilo; ali pa naj grem sam ponjo. Toda potlej ne vem? -- Ko bi mene več ne bilo nazaj? -- -- Bogu je vse mogoče!"
            
            Cesar, ko to sliši, urno pošlje na Vrh po Krpanovo kobilico. Ko jo pripeljejo na Dunaj, Krpan reče: "Zdaj pa le vkup dunajski junaki, kjer vsa je še kaj! Moje kobilice, kakor je videti slaba, vendar nihče ne potegne do praga, nikar že cez prag!"
            
            Skušali so jahači in konjarji in vsi tisti, ki so učeni, kako velja v strah prijeti konja, bodisi hud ali pa krotak, pa kobilice ni nihče premaknil z mesta; vsakega je vrgla na gnojno gomilo. "Bes te lopi!" reče eden in drug. "Majhno kljuse, velika moč!"
            
            Prišel je čas voja z velikanom; bilo je ravno svetega Erazma dan. Krpan vzame kij in mesarico, zasede kobilico, pa jezdi iz mesta na travnik, kjer se je Brdavs bojeval. Martina je bilo čudno gledati: njegova kobilica je bila majhna, noge je imel velike, tako da so se skoraj po tleh za njim vlekle; na glavi je nosil star klobuk širokih krajev, na sebi pa debelo suknjo iz domače volne; vendar se nobenega ni bal; celo sam cesar ga je rad poslušal, kadar je kakšno prav žaltavo razdrl.
            
            Ko ugleda Brdavs jezdeca, svojega sovražnika, začne s hrohotom smejati se in reče: " Ali je to tisti Krpan, ki so ga poklicali nadme tako daleč, tam z Vrha od Svete Trojice? Mar bi raje bil ostal doma za pečjo, da bi ne cvelil svoje stare matere, ako jo še imaš, da bi ne žalil svoje žene, ako ti jo je Bog dal. Pojdi mi izpred oči, da te videl ne bom, pa le naglo, dokler mi je srce še usmiljeno. Če me zgrabi jeza, ležal boš na zemlji krvav, kakor je sam cesarjev sin in sto drugih!"
            
            Krpan mu odgovori: "Če nisi z Bogom še spravljen, urno skleni, kar imaš; moja misel ni, dolgo čakati, mudi se mi domov za peč; tvoje besede so mi obudile v srcu živo željo do svoje koče in do svoje peči; ali poprej vendar ne pojdem, da tebi vzamem glavo. Pa ne zameri! To mi je naročil moje gospod, cesar; jaz nisem vedel ne zate ne za tvoje velikanstvo in za vse krvave poboje. Prijezdi bliže, da si podava roke; nikoli si jih nisva poprej; nikoli si jih ne bova pozneje; ali pravijo, da Bog nima rad, če pride kdo z jezo v srcu pred sodni stol."
            
            Velikan se nekoliko začudi, ko to sliši. Naglo prijezdi ter mu poda svojo debelo roko. Krpan mu jo pa tako stisne, da precej kri izza nohtov udari.
            
            Brdavs malo zareži, pa vendar nič ne pravi, ampak misli si: ta je hud in močan; pa kaj bo -- kmet je kmet; saj ne zna bojevati se, kakor gre junakom.
            
            Urno zasukneta vsak svojega konja in zdirjata si od daleč naproti. Brdavs visoko vzdigne meč, da bi že o prvem odsekal sovražniku glavo; ali ta mu urno podstavi svoj kij, da se meč globoko zadere v mehko lipovino; in preden ga velikan more izdreti, odjaha Krpan z male kobilice, potegne Brdavsa na tla, pa ga položi, kakor bi otroka v zibel deval, ter mu stopi za vrat in reče: "No, zdaj pa le hitro izmoli en očenašek ali dva in pa svojih grehov se malo pokesaj; izpovedal se ne boš več, nimam časa dolgo odlašati, mudi se mi domov za peč; znaj, da komaj čakam, da bi zopet slišal zvon, ki poje na Vrhu pri Sveti Trojici."
            
            To izreče, pa vzame počasi mesarico ter mu odseka glavo in se vrne proti mestu.
            
            Dunajčanje, ki so do zdaj le od daleč gledali, pridero k njemu, tudi sam cesar mu pride naproti, pa ga objame pričo ljudstva, ki je vpilo na vse grlo: "Krpan je nas otel! Hvala Krpanu, dokler bo Dunaj stal!"
            
            Krpanu se je to kaj dobro zdelo, da je dosegel toliko čast in držal se je na svoji kobilici, kakor bi šel v gostje vabit. Saj se je tudi lahko; še tu med nami, če kdo kakega slepca ali belouško ubije, še ne ve, na kateri grm bi jo obesil, da bi jo videlo več ljudi.
            
            Ko pridejo v cesarsko poslopje vsi knezi, vojskovodje in vsa prva gospoda s Krpanom, spregovori najprvo sam cesar in pravi: "Zdaj si pa le izberi! Dam ti, kar želiš, ker si zmogel tolikega sovražnika in otel deželo in mesto velike nadloge in nesreče. Nimam take stvari v cesarstvu, da bi dejal: ne dam ti je, če jo hočeš; celo Jerico, mojo edino hčer, imaš na ponudbo, ako nisi še oženjen."
            
            "Oženjen sem bil, pa nisem več," odgovori Krpan; "rajnica je umrla, druge pa nisem iskal. Sam ne vem, kako bi vendar, da bi ne bilo meni napak, Bogu in dobrim ljudem pa všeč. Vašega dekleta sem že videl. Če je tudi še tako pametna, kakor je lepa, potlej naj se le skrije moja babnica pred njo v vseh rečeh. Dobrote, res, da je navajena, tistega ne bom dejal, ker je od bogate hiše doma; pa saj na Vrhu pri Sveti Trojici spet nismo zgolj berači; pri nas tudi skozi vse leto visi kaj prekajenega na ražnju. Samo to ne vem, kako bo. -- Nesla sva bila z Marjeto v oprtnih košéh enkrat grozdje v Trst. Nazaj grede mi je bila pa ona zbolela na potu. Tako se mi je sitno zdelo, da vam na morem povedati! Raje bi bil imel, da bi se mi bili utrgali v cerkvi naramnici obe kmalu, takrat ko bi ravno bil sveče prižigal. Ni bilo drugače: naložil sem jo v oprtni koš, koš pa na pleči ter sem koračil mastito ž njo! Izhajal bi že bil; saj Mretačka je bila tako majhna kakor deklina trinajstih let -- pa jih je nadloga vendar imela že trideset, ko sva se jemala -- težka tedaj ni bila; ali kamor sem prišel, povsod so me vprašali, kakšno kramo prodajam. To je presneto slaba krama, babo po svetu prenašati! In ko bi se zdaj na cesti nama spet kaj takega nakretilo, vaši hčerki in pa meni? Od tukaj do Vrha se pot vleče kakor kurja čeva. Koša revež nimam, kobilica ima pa samo eno sedlo! Pa bi tudi ne bilo čudo, ki bi zbolela; saj vemo vsi, da take mehkote niso vajene od petih zjutraj do osmih zvečer cika coka, cika coka s konjem. Če se to prav do dobrega vse premisli, menda bo najbolje, da vam ostane cesarična, meni pa vdovstvo, čeravno pravzaprav dosti ne maram zanje; ali kar Bog dá, tega se človek ne sme braniti."
            
            Cesarica pa še zdaj ni bila pozabila košate lipe nad kamnito mizo na vrtu; zato je tudi ni bilo zraven, poslušala pa je za vrati, kakor imajo ženske navado, ki bi rade vse izvedele. Ko sliši, da cesar ponuja Krpanu svojo hčer v zakon, pride tudi ona in pravi: "Ne boš je imel, ne! Lipo si mi izpridil; hčere ti pa ne dam! Ljubeznivi moj mož, menda da ti je kri v glavi zavrela -- ne morem ti dobrega reči -- da govoriš take besede, ki sam dobro veš, da so prazne ena in druga. Pa tudi vas naj bo sram, vas, gospodje! Grdo je tako, da se mora kmet za vas bojevati! Še dandanes bi lipa lahko stala, pa tudi velikan več ne imel glave, ko bi vi kaj veljali. Pa saj vem: kar so se obabili možje, je vsaka baba neumna, katera se omoži! Res je, Krpan, otel si cesarstvo in tudi Dunaj si otel; zato boš pa dobil vina sod, ki drži petdeset malih veder, potem sto in pet pogač, dvajset janjcev in pa oseminštirideset krač ti bomo dali. Dobro me poslušaj! To moraš pa vse domu na Kranjsko spraviti, ako hočeš. Prodati pa ne smeš cepêra, ne tukaj ne na potu. Kadar boš na Vrhu pri Sveti Trojici, potlej pa stori, kakor se ti zdui. In ker zdaj nimamo tukaj nobenega Brdavsa več, menda ne bo napak, če osedlaš imenitno svojo kozico, ki praviš, da je kobilica, pa greš lepo počasi proti Vrhu. Pozdravi mi tamkaj vse Vrhovščake, posebno pa mater županjo!"
            
            Cesarica je to izgovorila, pa je šla precej spet v svoje hrame. Vseh gospodov je bilo jako sram. Kaj bi jih pa tudi ne bilo? Prebito jih je obrenkala; prav kakor takim gre! Krpan se je pa držal, da je bil skoraj hudemu vremenu podoben. Kakor bi se za Mokrcem bliskalo, tako je streljal z očmi izpod srditega čela; obrvi so mu pa sršele ko dve metli. Da te treni, kako je bilo vsem okoli njega čudno pri srcu! Še cesar je plaho od strani gledal, cesar! Pa vendar, ker sta bila vedno velika prijatelja, zato se počasi predrzne in reče mu: "Krpane, le ti molči; midva bova že naredila, da bo prav!"
            
            Krpan ga pa nič ne posluša, temveč zadene si na desno ramo kij, na levo pa mesarico, stopi k durim in reče: "Veste kaj? Bog vas obvari! Pa nikar kaj ne zamerite!"
            
            Na te besede prime za kljuko, pa kakor da bi hotel iti.
            
            Cesar poteče za njim: "Čaki no! Daj si dopovedati! Bog nas vari; saj menda nisi voda!"
            
            Krpan odgovori: "Koga? Menite, da nisem še zadosti slišal, ka-li? Meni bi gotovo segla brada že noter do pasa ali pe še dalj, ko bi se ne očedil vsak teden dvakrat; pa bo kdo pometal z mano? Kdo je pome poslal kočijo in štiri konje? Vi ali jaz? Dunaja ni bilo meni treba, mene pa Dunaju! Zdaj se pa takisto dela z mano? In pa še zavoljo mesa in vina moram očitke požirati? Že nekatere matere sem jel kruh, črnega in belega; nekaterega očeta vino sem pil: ali nikjer, tudi pri vas nisem in ne bom dobil take postrežbe, kakršna je v Razdrtem pri Klinčarju. Ni grjega na tem svetu kakor to, če se kaj da, potlej pa očita! Kdor noče dati, pa naj ima sam! Pa tudi, kdo bi mislil, da lipove pravde še zdaj ni kraja ne konca? Ali je bilo tisto drevesce vaš bog ali kaj? Tak les raste za vsakim grmom, Krpana pa ni za vsakim voglom, še na vsakem cesarskem dvoru ne, hvalo Bogu! Darove pa spet take dajate, da človek ne more do njih; to je ravno, kakor bi mački miš na rep privezal, da se potlej vrti sama okrog sebe, doseči je pa vendar ne more. Petdeset malih veder vina, pet in sto pogač, dvajset janjcev in pa oseminštirideset krač; tako blago res ni siromak; ali kaj pomaga! Prodati ne smem; z Dunaja na Vrh pa tudi ne kaže prenašati! Pa jaz bom drugo naredil, kakor se nikomur ne zdi! Deske si bom znesel na dvorišče in, ako jih bo premalo, potlej bo pa drevje zapelo. Vse bom posekal, kar mi bo prišlo pod sekiro, bodisi lipa ali pa lipec, hudolesovina ali dobroletovina, nad kamnito ali nad leseno mizo; pa bom postavil sredi dvorišča kolibo in tako dolgo bom ležal, dokler bo sod moker, pa dokler bom imel kaj prigrizniti. Ampak to vam pravim: samo še enkrat naj pride Brdavs na Dunaj, potlej pa zopet pošljite pome kočijo in služabnika, ali pa še celo svojo hčer, ki ne maram zanjo malo in dosti ne; pa bomo videli, kaj boste pripeljali z Vrha od Svete Trojice! Ako bo Krpan, mesa in kosti gotovo ne bo imel; ampak iz ovsene slame si ga boste morali natlačiti; pa se ga ne bodo še vrabci dolgo bali, nikar že velikani! Mislil sem iti brez besedice govorjenja. Ker ste me pa sami ustavili, ne bodite hudi, če sem vam katero grenko povedal; saj menda veste, kako je dejal rajnik Jernejko na Golem: 'Ali ga bom s pogačo pital, kadar se s kom kregam! Kar ga bolj ujezi, to mu zabelim.' zdaj pa le zdravi ostanite!"
            
            Cesar pravi na to: "Martin, potrpi no! Vsaj ne bodi tako neučakaven. Ti ne pojdeš od naše hiše, verjemi da ne! Saj sem jaz gospodar, veš!"
            
            Krpan odgovori: "Vsak človek je tak, kakršnega je Bog dal; vsak ima nekaj nad sabo: kdor ni grbast, morda pa je trobast! Moje obnašanje ni za vsa, že vidim, da ne. Tega se tedaj ne menimo, da bi jaz tukaj ostal. Saj tudi kobilica, ki se ji pravi kozica, ni vajena zmerom ob suhi krmi. Doma se je pasla po gozdu, na potu pa ob cestah!"
            
            Na to pristopi minister Gregor, ki je imel ključe od cesarske kase, ker taki imajo za vsako reč posebej služabnika. Minister se oglasi: "Cesarost, veste kaj? Vaš norec Stehàn je umrl; včeraj smo imeli osmi dan za njim, Bog mu daj nebeško luč! Stehan in Krpan, to si je nekam jako podobno. Kaj menite? Morda bi le-ta prevzel njegovo službo? Nič se ne ve. Zvitorepec je; debel je; smešen tudi, jezičen ravno tako; vse krščanstvo ga nima takega!"
            
            Krpan odgovori: "Magister Gregor, veste kaj? Enkrat sem bil vaš bebec, dvakrat pa ne bom. Smejalo bi se malo in veliko meni in moji zarobljeni pameti, ko bi to naredil. -- Zdaj pa dobro, da mi je prišlo na misel! Kmalu bi bil pozabil, kar imam že dolgo na jeziku. Cesarost, nekdaj ste me bili srečali s kobilico v snegu, kajne?"
            
            Cesar: "Bilo je tako, bilo!"
            
            Krpan: "Kaj pa sem nesel na tovoru?"
            
            Cesar: "Bruse pa kresilno gobo."
            
            Krpan: "To je bilo tačas, ko ste se vi peljali v Jeruzalem."
            
            Cesar: "Bosa je ta! V Trst sem šel; za Jeruzalem toliko vem, kakor za svojo zadnjo uro."
            
            Krpan: "Jaz pa za bruse in kresilno gobo ravno toliko. Takrat, veste, vam nisem bil resnice povedal, kar mi je še zdaj žal. Angleško sol sem prenašal. Saj se nisem bal pravzaprav ne vas ne vašega kočijaža. Pa taka je le: kadar se človek zasukne s pravega pota, naj bo še tako močan, pa se vendar boji, če veja ob vejo udari."
            
            Na to pravi minister Gregor: "Ne veš, da je prepovedano? To je nevaren človek; državi dela škodo. Primite ga, zaprimo ga!"
            
            Krpan odgovori: "Kdo me bo? Morda vi dolgopetec, ki ste suhi kakor raženj; ki je vas in vašega magistrovanja z vami komaj za polno pest? Z eno samo roko vas porinem čez svetega Štefana streho, ki stoji sredi mesta! Nikar praznih besed ne razdirajte!"
            
            Cesar pravi: "Le ti meni povedi, če bi morda še kaj rad. Midva ne bova v sovraštvu ostala, ne, če Bog da, da ne. Minister Gregor, ti ga pa le pusti! Že jaz vem, kako je!"
            
            Krpan odgovori: "Poslušajte me tedaj! Moje otepanje z Brdavsom vem, da je imena vredno. Kaj se zna? Morda bodo postavači še celo skladali pripovedovavke in pesmi, da se bo govorilo, ko ne bo ne vas ne mene kosti ne prsti, če ne bo magister Gregor dal drugače v bukve zapisati. Pa naj stori, kakor če; meni se ne bo s tem ne prikupil ne odkupil. Ampak vendar je vsak delavec vreden svojega plačila, to sem v cerkvi slišal. Če je vaša draga volja, dajte mi tedaj pismo, ki bo veljavno pred vsako duhovsko in deželsko gosposko; pa tudi svoj pečat morate udariti, da bom brez skrbi nosil angleško sol po svetu. Če mi to daste, naj bom ves malopridnež, kolikor me je pod klobukom, ako vam bom kdaj kaj opotikal, dokler bom tovoril!"
            
            Cesar je bil precej pri volji; minister Gregor pa nikakor ni pritegnil. Ali cesar ga ni poslušal, ampak šele dejal je: "Gregor, vzemi pero, pa zapiši, kakor je Martin rekel!"
            
            Minister Gregor se je kislo držal, branil se pa le ni, kar so mu veleli; zakaj cesarja se vendar vsak boji. Kadar je bilo pismo narejeno in zapečateno, pravi cesar Krpanu: "Martin, ali prodaš meni pogače in vino, pa kar je še drugih reči? Najlaže bo tako, lej! S cesarico bom že jaz govoril, da bo prav. Mošnjo cekinov ti dam; ti boš pa blago pustil. Kdo bo to prenašal z Dunaja do Svete Trojice?"
            
            Krpan odgovori: "Poldrugo mošnjo pa še kakšno krono povrhu, vem, da je lepo vredno, ko bi prodajal brat bratu. Pa naj bo, no, pri vas ne bom na tisto gledal. Samo da jaz ne bom imel pri cesarici zavoljo tega nikakršnih ohodkov; ne lazim rad okoli gosposke! Pa saj imam priče, da ste vi prevzeli vse sitnosti, ki bodo prišle prvič ali drugič iz tega, dobro me poslušajte!"
            
            Cesar mu dé: "Nič se ne boj; to bom že poravnal sam brez tebe. Ná mošnjo; tu je pa še pismo. Saj nocoj tako še ne pojdeš iz grada, če le misliš iti; priklonil se je že dan ter noč se bliža."
            
            Ali Krpan odgovori: "Lepa hvala vam bodi najpopred za pisemce, da ga bom v zobe vrgel vsakemu, kdor me bo ustavljal na cesti; pa tudi zavoljo mošnjička se ne bom krčil. Kaj se ve, kaj zadene človeka v neznanju? Morda mi utegne še na hvalo priti. Vselej pravijo: bolje drži ga, kakor lovi ga! Pri vas pa ne bom ostajal čez noč, ako se vam ne zamerim skozi to. Že hudo me ima, da bi spet enkrat bil na Vrhu pri Sveti Trojici. Samo še nekaj bi vas rad prosil, ko bi mi dali človeka, da bi me spremil do ceste. Mesto je veliko; hiš je, kolikor jih še nisem videl, kar sol prenašam, akoravno sem že na Reki bil, tudi v Kopru, na Vrhniki in v Ljubljani; ali tolikih ulic ni nikjer. S kočijažem sva se hitro vozila in toliko vem, kod sem prišel, kakor bi bil imel oči zavezane; pa sem vendar gledal na levo in tudi na desno; ali to ni dano vsakemu človeku, da bi vselej vedel, kje je."
            
            Cesar mu je obljubil svojega služabnika, potlej mu je roko podal, pa tudi Gregorju velel, da naj mu roko poda. Minister se ni branil; ali vendar je bil zavoljo pisma ves zelen od jeze.
            
            Krpan zadene kij in mesarico, in to so bile njegove zadnje besede pred cesarjem: "Ko bi se spet oglasil kak Brdavs ali kdo drug, saj veste, kje se pravi na Vrhu pri Sveti Trojici. Pozdravil bom pa že Vrhovščake in mater županjo. Zdravi ostanite!"
            
            "Srečno hodi!" pravi cesar, minister Gregor pa nič."""
            
            _odlomek = """V Notranjem stoji vas, Vrh po imenu. V tej vasici je
            #     živel v starih časih Krpan, močan in silen človek. Bil je neki
            #     tolik, da ga ni kmalu takega. Dela mu ni bilo mar; ampak nosil je
            #     od morja na svoji kobilici angleško sol, kar je bilo pa že tistikrat
            #     ostro prepovedano. Pazili so ga mejači, da bi ga kje nehotoma zalezli;
            #     poštenega boja ž njim so se bali ravno tako kakor pozneje Štempiharja.
            #     Krpan se je pa vedno umikal in gledal, da mu niso mogli do živega."""
            
            test_data = [
                ("""najdi_besede(odlomek, 'il')""",
                 {'silen', 'bilo', 'Bil', 'nosil', 'Pazili', 'kobilici'}),
                ("""najdi_besede(odlomek, 'so')""",
                 {'sol', 'niso', 'so'}),
                ("""najdi_besede(martin_krpan_z_vrha, 'sil')""",
                 {'prosil', 'silni', 'kresilno', 'skusil', 'oglasil', 'silo', 'obesil', 'Kresilno', 'silen', 'silnega', 'nosil'}),
                ("""najdi_besede(martin_krpan_z_vrha, 'čen')""",
                 {'jezičen', 'naročeno', 'počenil', 'učeni', 'nevšečen', 'očenašek'}),
                ("""najdi_besede(martin_krpan_z_vrha, 'kov')""",
                 {'nakovalo', 'kovač', 'vojskovodje', 'kovačnico', 'kovačnica', 'junakov', 'ohodkov', 'obrekovalci', 'vojščakov'}),
            ]
            for td in test_data:
                if not Check.equal(*td, env={'odlomek': _odlomek,
                                             'martin_krpan_z_vrha': _martin_krpan_z_vrha}):
                    break
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTA2LCJ1c2VyIjo4MDYxfQ:1s4eJr:RL5k2WTcGZGwTYZuN98vkIuQuNwKk3aE_DVEU0iG2J8"
        try:
            test_data = [
                ("""najdi_predpono(odlomek, 'po')""", {'pozneje', 'poštenega', 'po'}),
                ("""najdi_predpono(odlomek, 'če')""", set()),
                ("""najdi_predpono(martin_krpan_z_vrha, 'pred')""", {'predlansko', 'preden', 'predrzne', 'pred'}),
                ("""najdi_predpono(martin_krpan_z_vrha, 'poz')""",
                 {'poznal', 'pozabil', 'poznaš', 'pozabila', 'pozneje', 'pozimi'}),
                ("""najdi_predpono(martin_krpan_z_vrha, 'kon')""",
                 {'konca', 'konják', 'konjarji', 'konje', 'koncu', 'konjiča', 'konjem', 'konja', 'konj', 'konji'}),
            ]
            for td in test_data:
                if not Check.equal(*td, env={'odlomek': _odlomek,
                                             'martin_krpan_z_vrha': _martin_krpan_z_vrha}):
                    break
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTA3LCJ1c2VyIjo4MDYxfQ:1s4eJr:2BW7Owt74NkkMu_SffqcgXC_Lpktuq7PStlGq0NjKJM"
        try:
            test_data = [
                ("""najdi_pripono(odlomek, 'ga')""", {'takega', 'ga', 'poštenega', 'živega'}),
                ("""najdi_pripono(odlomek, 'čen')""", set()),
                ("""najdi_pripono(martin_krpan_z_vrha, 'ski')""", {'cesarski', 'pravdanski', 'dunajski'}),
                ("""najdi_pripono(martin_krpan_z_vrha, 'ili')""",
                 {'smili', 'Pazili', 'kobili', 'veselili', 'vodili', 'hodili', 'obabili', 'bili', 'ustavili', 'lazili'}),
                ("""najdi_pripono(martin_krpan_z_vrha, 'ec')""",
                 {'Zvitorepec', 'bebec', 'dolgopetec', 'delavec', 'norec', 'lipec'}),
            ]
            for td in test_data:
                if not Check.equal(*td, env={'odlomek': _odlomek,
                                             'martin_krpan_z_vrha': _martin_krpan_z_vrha}):
                    break
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTA4LCJ1c2VyIjo4MDYxfQ:1s4eJr:qtyLFf8fNB0GRj-ly-O4GnL1WxE4g4gIY4OVWQZUaI0"
        try:
            Check.equal
            test_data = [
                ("""podvojene_crke('A volunteer is worth twenty pressed men.')""", {'pressed', 'volunteer'}),
                ("""podvojene_crke(martin_krpan_z_vrha)""", {'izza'}),
                ("""podvojene_crke('Mississippi, I will remember you\\nwhenever I should go away\\nI will be longing for the day\\nthat I will be in Greenville again')""",
                 {'Greenville', 'Mississippi', 'will'}),
            ]
            for td in test_data:
                if not Check.equal(*td, env={'odlomek': _odlomek,
                                             'martin_krpan_z_vrha': _martin_krpan_z_vrha}):
                    break
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    print("Shranjujem rešitve na strežnik... ", end="")
    try:
        url = "https://www.projekt-tomo.si/api/attempts/submit/"
        token = "Token 80dacd4c4c892bbbdfac41ee2f24fc653c671c34"
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        message = (
            "\n"
            "-------------------------------------------------------------------\n"
            "PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE!\n"
            "Preberite napako in poskusite znova ali se posvetujte z asistentom.\n"
            "-------------------------------------------------------------------\n"
        )
        print(message)
        traceback.print_exc()
        print(message)
        sys.exit(1)
    else:
        print("Rešitve so shranjene.")
        update_attempts(Check.parts, response)
        if "update" in response:
            print("Updating file... ", end="")
            backup_filename = backup(filename)
            with open(__file__, "w", encoding="utf-8") as f:
                f.write(response["update"])
            print("Previous file has been renamed to {0}.".format(backup_filename))
            print("If the file did not refresh in your editor, close and reopen it.")
    Check.summarize()


if __name__ == "__main__":
    _validate_current_file()
