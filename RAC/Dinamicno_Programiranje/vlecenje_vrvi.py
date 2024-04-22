# =============================================================================
# Vlečenje vrvi
#
# Udeleženci piknika bodo vlekli vrv. So različnih spolov, starosti in
# mas, zato sprva niso vedeli, kako bi se pravično razdelili v dve
# skupini. Sklenili so, da je najpravičnejša razdelitev takšna, da bosta
# imeli obe skupini enako skupno težo, na število članov skupin pa se
# sploh ne bodo ozirali. Včasih dveh skupin s popolnoma enakima masama
# ni mogoče sestaviti, zato iščejo takšno razdelitev, da bo razlika med
# masama skupin čim manjša. Vsak udeleženec nam je zaupal svojo  maso v
# kilogramih in sicer jo je zaokrožil na najbližje celo število.
# =====================================================================@005831=
# 1. podnaloga
# Sestavite funkcijo `razdeli(mase)`, ki dobi seznam mas udeležencev in vrne skupno maso
# manjše od skupin pri najbolj pravični delitvi. Ta funkcija naj deluje *s sestopanjem*,
# torej pregleda vse možnosti. (Katere so vse možnosti in koliko jih je?) Zgled, v katerem
# je optimalna razdelitev dosežena, ko damo udeleženca z masama 102 in 95 skupaj eno
# skupino, vse ostale pa v drugo:
# 
#     >>> razdeli([95, 82, 87, 102, 75])
#     197
# 
# *Navodilo*: naj bo `skupaj` skupna masa vseh udeležencev, torej vsota števil v seznamu
# `mase`. Definirajmo pomožno funkcijo `deli(levi, i)`, ki sprejme maso `levi`
# udeležencev, ki so bili do sedaj razporejeni v levo skupino, ter indeks `i` naslednjega
# udeleženca, ki ga moramo še razporediti. Funkcija `razdeli` potem enostavno pokliče `deli(0,0)`.
# 
# Funkcija `deli` je rekurzivna in pregleduje vse možnosti. Pri tem pazi, da masa `levi`
# ne preseže `skupaj//2`, da se izogne dvakratnemu pregledovanju simetričnih kombinacij.
# Funkcija vrne maso lažje od obeh skupin:
# 
# * če je `i == len(mase)`, smo obravnavli vse, in vrnemo `levi`
# * če bi dali `i`-tega na levo in bi s tem leva skupina presegla `skupaj//2`, potem ga
#   damo na desno
# * sicer rekurzivno preizkusimo obe možnosti (`i`-tega damo na levo ali na desno)
#   ter se odločimo za boljšo od obeh možnosti.
# =============================================================================
def razdeli(mase):
    

# =====================================================================@005832=
# 2. podnaloga
# Če zgornjo rešitev preizkusite na seznamih dolžine 25 ali več, boste
# ugotovili, da deluje izjemno počasi. Kakšna je njena časovna zahtevnost?
# 
# Nalogo bomo rešili še z dinamičnim programiranjem. Gre za tako imenovani
# _problem 0-1 nahrbtnika_. Izkoristili bomo dejstvo, da mase ljudi ne
# morejo biti poljubno velike (največja dokumentirana masa človeka je 635 kg)
# in da so celoštevilske. Pri sestavljanju skupin lahko dosežemo enako
# maso na različne načine.
# 
# Sestavite funkcijo `razdeli_dinamicno(mase)`, ki naredi isto kot prejšnja
# funkcija, le da se reševanja tokrat lotite z dinamičnim programiranjem.
# Zgled:
# 
#     >>> razdeli_dinamicno([95, 82, 87, 102, 75])
#     197
# 
# Funkcijo preizkusite na seznamu dolžine 50 in na seznamu dolžine 100.
# 
# *Navodilo:* Naj bo `skupaj` skupna masa vseh udeležencev. Tokrat bomo izračunali množico
# `mozna` tako da bo veljalo `i ∈ mozna` natanko tedaj, ko je možno razdeliti udeležence
# tako, da ima ena od obeh skupin maso `i`. To lahko naredimo s preprosto zanko,
# upoštevajoč:
# 
# * `0 ∈ mozna`, če damo vse udeležence v eno skupino
# *  če imamo udeleženca z maso `k`, ki ga še nismo obravnavali, in je `i ∈ mozna`, potem velja tudi
#    `(i+k) ∈ mozna`.
# 
# Ko enkrat imamo tabelo `mozna`, poisčemo največji indeks `i`, ki je manjši ali enak
# `skupaj//2` in je `mozna[i] == True`.
# =============================================================================

# =====================================================================@005833=
# 3. podnaloga
# Prejšnja funkcija nam izračuna maso manjše skupine, nič pa ne izvemo o tem,
# kdo so udeleženci, ki tvorijo to skupino. Sestavite še funkcijo
# `razdeli_udelezence(mase)`, ki vrne seznam mas udeležencev, ki sestavljajo
# manjšo od obeh skupin. Če je rešitev več, lahko funkcija vrne
# katerekoli rešitev. Zgled:
# 
#     >>> razdeli_udelezence([95, 82, 87, 102, 75])
#     [102, 95]
# 
# _Namig:_ Spremenite prejšnjo rešitev tako, da bo namesto množice možnih mas skupin `mozna`
# izračunala slovar `skupine`. Ključi v slovarju so možne mase (torej elementi množice
# `mozna`), vrednosti pa množice, ki sestavljajo pripadajočo skupino.
# =============================================================================






































































































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
        ] = "eyJwYXJ0Ijo1ODMxLCJ1c2VyIjo4MDYxfQ:1ruUz3:3ZaI4E6temQw_1dvEAMjYh0N9jXjT1301GKVew-P8Nc"
        try:
            Check.equal("""razdeli([95, 82, 87, 102, 75])""", 197)
            Check.equal("""razdeli([60, 120])""", 60)
            Check.equal("""razdeli([80])""", 0)
            Check.equal("""razdeli([])""", 0)
            Check.equal("""razdeli([73, 91, 73, 80, 105, 71, 82, 102, 91, 76])""", 422)
            Check.equal("""razdeli([89, 86, 76, 101, 74, 109, 119, 102, 87, 111, 107, 115])""", 588)
            Check.equal("""razdeli([103, 80, 91, 120, 100, 106, 117, 119, 70, 82, 107, 85, 85])""", 632)
            Check.equal("""razdeli([114, 91, 72, 84, 97, 114, 97, 119, 106, 81, 86, 98, 72, 83])""", 657)
            Check.equal("""razdeli([115, 120, 89, 105, 102, 100, 101, 98, 106, 104, 86, 90, 100, 70, 74])""", 730)
            Check.equal("""razdeli([119, 72, 80, 111, 87, 84, 111, 76, 104, 73, 90, 78, 112, 82, 105, 76, 100, 98])""", 829)
            Check.equal("""razdeli([92, 89, 83, 91, 87, 110, 119, 119, 89, 96, 113, 82, 79, 97, 114, 84, 70, 90, 97])""", 900)
            Check.equal("""razdeli([98, 99, 103, 72, 117, 88, 93, 70, 78, 90, 104, 96, 101, 79, 119, 105, 107, 109, 71, 93])""", 946)
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
        ] = "eyJwYXJ0Ijo1ODMyLCJ1c2VyIjo4MDYxfQ:1ruUz3:FMNlUaYB8VGjCu0O3yuFHifV3BimP-_bQfWDpfSYRqQ"
        try:
            Check.equal("""razdeli_dinamicno([95, 82, 87, 102, 75])""", 197)
            Check.equal("""razdeli_dinamicno([60, 120])""", 60)
            Check.equal("""razdeli_dinamicno([80])""", 0)
            Check.equal("""razdeli_dinamicno([])""", 0)
            Check.equal("""razdeli_dinamicno([73, 91, 73, 80, 105, 71, 82, 102, 91, 76])""", 422)
            Check.equal("""razdeli_dinamicno([89, 86, 76, 101, 74, 109, 119, 102, 87, 111, 107, 115])""", 588)
            Check.equal("""razdeli_dinamicno([103, 80, 91, 120, 100, 106, 117, 119, 70, 82, 107, 85, 85])""", 632)
            Check.equal("""razdeli_dinamicno([114, 91, 72, 84, 97, 114, 97, 119, 106, 81, 86, 98, 72, 83])""", 657)
            Check.equal("""razdeli_dinamicno([115, 120, 89, 105, 102, 100, 101, 98, 106, 104, 86, 90, 100, 70, 74])""", 730)
            Check.equal("""razdeli_dinamicno([119, 72, 80, 111, 87, 84, 111, 76, 104, 73, 90, 78, 112, 82, 105, 76, 100, 98])""", 829)
            Check.equal("""razdeli_dinamicno([92, 89, 83, 91, 87, 110, 119, 119, 89, 96, 113, 82, 79, 97, 114, 84, 70, 90, 97])""", 900)
            Check.equal("""razdeli_dinamicno([98, 99, 103, 72, 117, 88, 93, 70, 78, 90, 104, 96, 101, 79, 119, 105, 107, 109, 71, 93])""", 946)
            Check.equal("""razdeli_dinamicno([70, 78, 88, 80, 101, 109, 102, 97, 70, 114, 79, 77, 103, 81, 95, 118, 85, 82, 106, 94, 117, 114, 87, 104, 101, 112, 84, 95, 73, 120, 71, 120, 89, 74, 80, 76, 115, 118, 110, 115, 74, 117, 95, 103, 119, 99, 105, 100, 88, 101])""", 2402)
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
        ] = "eyJwYXJ0Ijo1ODMzLCJ1c2VyIjo4MDYxfQ:1ruUz3:DyWRVm501hJY6nCkrTOndr72xW62Ywm_PYVv_atqfZc"
        try:
            test_data = [
                ([95, 82, 87, 102, 75], 197),
                ([60, 120], 60),
                ([80], 0),
                ([], 0),
                ([73, 91, 73, 80, 105, 71, 82, 102, 91, 76], 422),
                ([89, 86, 76, 101, 74, 109, 119, 102, 87, 111, 107, 115], 588),
                ([103, 80, 91, 120, 100, 106, 117, 119, 70, 82, 107, 85, 85], 632),
                ([114, 91, 72, 84, 97, 114, 97, 119, 106, 81, 86, 98, 72, 83], 657),
                ([115, 120, 89, 105, 102, 100, 101, 98, 106, 104, 86, 90, 100, 70, 74], 730),
                ([119, 72, 80, 111, 87, 84, 111, 76, 104, 73, 90, 78, 112, 82, 105, 76, 100, 98], 829),
                ([92, 89, 83, 91, 87, 110, 119, 119, 89, 96, 113, 82, 79, 97, 114, 84, 70, 90, 97], 900),
                ([98, 99, 103, 72, 117, 88, 93, 70, 78, 90, 104, 96, 101, 79, 119, 105, 107, 109, 71, 93], 946),
                ([70, 78, 88, 80, 101, 109, 102, 97, 70, 114, 79, 77, 103, 81, 95, 118, 85, 82, 106, 94, 117, 114, 87, 104, 101, 112, 84, 95, 73, 120, 71, 120, 89, 74, 80, 76, 115, 118, 110, 115, 74, 117, 95, 103, 119, 99, 105, 100, 88, 101], 2402),
            ]
            
            for udelezenci, masa in test_data:
                Check.equal("""all([x in {0} for x in razdeli_udelezence({0})])""".format(udelezenci), True)
                Check.equal("""sum(razdeli_udelezence({0})) == {1}""".format(udelezenci, masa), True)
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
