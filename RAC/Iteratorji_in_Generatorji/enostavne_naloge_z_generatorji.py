# =============================================================================
# Enostavne naloge z generatorji
# =====================================================================@005900=
# 1. podnaloga
# Sestavite generator `stevke(n)`, ki generira števke naravnega števila
# `n` začenši z enicami (s skrajno desno števko).
# 
#     >>> [x for x in stevke(1337)]
#     [7, 3, 3, 1]
# =============================================================================
def stevke(n):
    while n > 0:
        yield n % 10
        n = n // 10
# =====================================================================@005901=
# 2. podnaloga
# Sestavite generator `delitelji(n)`, ki kot argument dobi naravno število
# `n` in generira njegove delitelje (od najmanjšega proti največjemu).
# 
#     >>> [x for x in delitelji(12)]
#     [1, 2, 3, 4, 6, 12]
# =============================================================================
def delitelji(n):
    i = 1
    vecji = []
    while i**2 <= n:
        if n % i == 0:
            yield i
            if i**2 != n:
                vecji.append(n // i)
        i += 1
    for i in vecji[::-1]:
        yield i
# =====================================================================@005902=
# 3. podnaloga
# Sestavite generator `potence_naravnih(k)`, ki kot argument dobi število
# `k` in generira (neskončno) zaporedje $1^k, 2^k, 3^k, 4^k, \ldots$
# 
#     >>> g = potence_naravnih(2)
#     >>> [g.__next__() for i in range(10)]
#     [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
# =============================================================================
def potence_naravnih(k):
    i = 1
    while True:
        yield i**k
        i += 1

# =====================================================================@005903=
# 4. podnaloga
# Sestavite generator `fakultete(n)`, ki kot argument dobi nenegativno
# celo število `n` in generira (neskončno) zaporedje $n!, (n+1)!, (n+2)!, \ldots$
# 
#     >>> g = fakultete(0)
#     >>> [g.__next__() for i in range(10)]
#     [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
# =============================================================================
import math as m
def fakultete(n):
    prejsna = m.factorial(n)
    yield prejsna
    while True:
        n += 1
        yield prejsna * n
        prejsna = prejsna * n
# =====================================================================@005904=
# 5. podnaloga
# Spomnimo se, da je Collatzovo zaporedje z začetnim členom $a_0$ definirano
# s predpisom $a_{i+1} = a_i / 2$, če je $a_i$ sodo, in $a_{i+1} = 3 a_i + 1$ sicer.
# 
# Sestavite generator `collatz(n)`, ki kot argument dobi naravno število `n`
# in generira Collatzovo zaporedje z začetnim členom `n`. Generiranje naj se
# konča, ko pridemo do števila 1.
# 
#     >>> [x for x in collatz(20)]
#     [20, 10, 5, 16, 8, 4, 2, 1]
# =============================================================================
def collatz(n):
    yield n
    while True:
        if n == 1:
            break
        
        if n % 2 == 0:
            yield n / 2
            n = n / 2
        
        else:
            yield 3 * n + 1
            n = 3 * n + 1


    
    





































































































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
        ] = "eyJwYXJ0Ijo1OTAwLCJ1c2VyIjo4MDYxfQ:1rzZfl:TQFhD0e9Fh9B0ItzWD0FrfHBqhv2K6CrP7JjuPxFBCk"
        try:
            testCases = [("stevke(8)", [8], {'should_stop': True}),
                         ("stevke(43)", [3, 4], {'should_stop': True}),
                         ("stevke(162)", [2, 6, 1], {'should_stop': True}),
                         ("stevke(1337)", [7, 3, 3, 1], {'should_stop': True}),
                         ("stevke(526481379)", [9, 7, 3, 1, 8, 4, 6, 2, 5], {'should_stop': True})]
            for example, correct, options in testCases:
                if not Check.generator(example, correct, **options):
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
        ] = "eyJwYXJ0Ijo1OTAxLCJ1c2VyIjo4MDYxfQ:1rzZfl:SigN9TGBIRcqbGWX0KKWkBTI9SogLFqz4DoS3Dkzsdw"
        try:
            testCases = [("delitelji(7)", [1, 7], {'should_stop': True}),
                         ("delitelji(12)", [1, 2, 3, 4, 6, 12], {'should_stop': True}),
                         ("delitelji(25)", [1, 5, 25], {'should_stop': True}),
                         ("delitelji(60)", [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60], {'should_stop': True}),
                         ("delitelji(1001)", [1, 7, 11, 13, 77, 91, 143, 1001], {'should_stop': True}),
                         ("delitelji(2017)", [1, 2017], {'should_stop': True}),
                         ("delitelji(244140625)", [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125, 244140625], {'should_stop': True}),
                         ("delitelji(1220703125)", [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125, 244140625, 1220703125], {'should_stop': True}),
                         ("delitelji(1555200)", [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 256, 270, 288, 300, 320, 324, 360, 384, 400, 405, 432, 450, 480, 486, 540, 576, 600, 640, 648, 675, 720, 768, 800, 810, 864, 900, 960, 972, 1080, 1152, 1200, 1215, 1280, 1296, 1350, 1440, 1600, 1620, 1728, 1800, 1920, 1944, 2025, 2160, 2304, 2400, 2430, 2592, 2700, 2880, 3200, 3240, 3456, 3600, 3840, 3888, 4050, 4320, 4800, 4860, 5184, 5400, 5760, 6075, 6400, 6480, 6912, 7200, 7776, 8100, 8640, 9600, 9720, 10368, 10800, 11520, 12150, 12960, 14400, 15552, 16200, 17280, 19200, 19440, 20736, 21600, 24300, 25920, 28800, 31104, 32400, 34560, 38880, 43200, 48600, 51840, 57600, 62208, 64800, 77760, 86400, 97200, 103680, 129600, 155520, 172800, 194400, 259200, 311040, 388800, 518400, 777600, 1555200], {'should_stop': True}),
                         ("delitelji(1220703131)", [1, 1220703131], {'should_stop': True})]
            for example, correct, options in testCases:
                if not Check.generator(example, correct, **options):
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
        ] = "eyJwYXJ0Ijo1OTAyLCJ1c2VyIjo4MDYxfQ:1rzZfl:2Xe-V5lcaE2EI2fLRkR5k5MIOYJvHKhCDV3B3Jj6YYs"
        try:
            testCases = [("potence_naravnih(2)", [1, 4, 9, 16, 25, 36, 49, 64, 81, 100], {'further_iter': 100}),
                         ("potence_naravnih(3)", [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000], {'further_iter': 100}),
                         ("potence_naravnih(-1)", [1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666, 0.14285714285714285, 0.125, 0.1111111111111111, 0.1], {'further_iter': 100}),
                         ("potence_naravnih(0.5)", [1.0, 1.4142135623730951, 1.7320508075688772, 2.0, 2.23606797749979, 2.449489742783178, 2.6457513110645907, 2.8284271247461903, 3.0, 3.1622776601683795], {'further_iter': 100}),
                         ("potence_naravnih(5)", [1, 32, 243, 1024, 3125, 7776, 16807, 32768, 59049, 100000, 161051, 248832, 371293, 537824, 759375, 1048576, 1419857, 1889568, 2476099, 3200000],  {'further_iter': 100})]
            for example, correct, options in testCases:
                if not Check.generator(example, correct, **options):
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
        ] = "eyJwYXJ0Ijo1OTAzLCJ1c2VyIjo4MDYxfQ:1rzZfl:V3VFT2wGUZbYqtqLS2ggxbt1FW16ky_GxR-BPZIVPSQ"
        try:
            testCases = [("fakultete(0)", [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000], {'further_iter': 10}),
                         ("fakultete(1)", [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800], {'further_iter': 10}),
                         ("fakultete(2)", [2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800], {'further_iter': 10}),
                         ("fakultete(20)", [2432902008176640000, 51090942171709440000, 1124000727777607680000, 25852016738884976640000, 620448401733239439360000, 15511210043330985984000000], {'further_iter': 10}),
                         ("fakultete(10)", [3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000], {'further_iter': 500})]
            for example, correct, options in testCases:
                if not Check.generator(example, correct, **options):
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
        ] = "eyJwYXJ0Ijo1OTA0LCJ1c2VyIjo4MDYxfQ:1rzZfl:fsOXhIivx_Gl24O7cJRfzNjJHjQ9Nshg0siWk8NYZkQ"
        try:
            testCases = [("collatz(20)", [20, 10, 5, 16, 8, 4, 2, 1], {'should_stop': True}),
                         ("collatz(1)", [1], {'should_stop': True}),
                         ("collatz(97)", [97, 292, 146, 73, 220, 110, 55, 166, 83, 250, 125, 376, 188, 94, 47, 142, 71, 214, 107, 322, 161, 484, 242, 121, 364, 182, 91, 274, 137, 412, 206, 103, 310, 155, 466, 233, 700, 350, 175, 526, 263, 790, 395, 1186, 593, 1780, 890, 445, 1336, 668, 334, 167, 502, 251, 754, 377, 1132, 566, 283, 850, 425, 1276, 638, 319, 958, 479, 1438, 719, 2158, 1079, 3238, 1619, 4858, 2429, 7288, 3644, 1822, 911, 2734, 1367, 4102, 2051, 6154, 3077, 9232, 4616, 2308, 1154, 577, 1732, 866, 433, 1300, 650, 325, 976, 488, 244, 122, 61, 184, 92, 46, 23, 70, 35, 106, 53, 160, 80, 40, 20, 10, 5, 16, 8, 4, 2, 1], {'should_stop': True}),
                         ("collatz(73)", [73, 220, 110, 55, 166, 83, 250, 125, 376, 188, 94, 47, 142, 71, 214, 107, 322, 161, 484, 242, 121, 364, 182, 91, 274, 137, 412, 206, 103, 310, 155, 466, 233, 700, 350, 175, 526, 263, 790, 395, 1186, 593, 1780, 890, 445, 1336, 668, 334, 167, 502, 251, 754, 377, 1132, 566, 283, 850, 425, 1276, 638, 319, 958, 479, 1438, 719, 2158, 1079, 3238, 1619, 4858, 2429, 7288, 3644, 1822, 911, 2734, 1367, 4102, 2051, 6154, 3077, 9232, 4616, 2308, 1154, 577, 1732, 866, 433, 1300, 650, 325, 976, 488, 244, 122, 61, 184, 92, 46, 23, 70, 35, 106, 53, 160, 80, 40, 20, 10, 5, 16, 8, 4, 2, 1], {'should_stop': True}),
                         ("collatz(54)", [54, 27, 82, 41, 124, 62, 31, 94, 47, 142, 71, 214, 107, 322, 161, 484, 242, 121, 364, 182, 91, 274, 137, 412, 206, 103, 310, 155, 466, 233, 700, 350, 175, 526, 263, 790, 395, 1186, 593, 1780, 890, 445, 1336, 668, 334, 167, 502, 251, 754, 377, 1132, 566, 283, 850, 425, 1276, 638, 319, 958, 479, 1438, 719, 2158, 1079, 3238, 1619, 4858, 2429, 7288, 3644, 1822, 911, 2734, 1367, 4102, 2051, 6154, 3077, 9232, 4616, 2308, 1154, 577, 1732, 866, 433, 1300, 650, 325, 976, 488, 244, 122, 61, 184, 92, 46, 23, 70, 35, 106, 53, 160, 80, 40, 20, 10, 5, 16, 8, 4, 2, 1], {'should_stop': True}),
                         ("collatz(871)", [871, 2614, 1307, 3922, 1961, 5884, 2942, 1471, 4414, 2207, 6622, 3311, 9934, 4967, 14902, 7451, 22354, 11177, 33532, 16766, 8383, 25150, 12575, 37726, 18863, 56590, 28295, 84886, 42443, 127330, 63665, 190996, 95498, 47749, 143248, 71624, 35812, 17906, 8953, 26860, 13430, 6715, 20146, 10073, 30220, 15110, 7555, 22666, 11333, 34000, 17000, 8500, 4250, 2125, 6376, 3188, 1594, 797, 2392, 1196, 598, 299, 898, 449, 1348, 674, 337, 1012, 506, 253, 760, 380, 190, 95, 286, 143, 430, 215, 646, 323, 970, 485, 1456, 728, 364, 182, 91, 274, 137, 412, 206, 103, 310, 155, 466, 233, 700, 350, 175, 526, 263, 790, 395, 1186, 593, 1780, 890, 445, 1336, 668, 334, 167, 502, 251, 754, 377, 1132, 566, 283, 850, 425, 1276, 638, 319, 958, 479, 1438, 719, 2158, 1079, 3238, 1619, 4858, 2429, 7288, 3644, 1822, 911, 2734, 1367, 4102, 2051, 6154, 3077, 9232, 4616, 2308, 1154, 577, 1732, 866, 433, 1300, 650, 325, 976, 488, 244, 122, 61, 184, 92, 46, 23, 70, 35, 106, 53, 160, 80, 40, 20, 10, 5, 16, 8, 4, 2, 1]
            , {'should_stop': True}),
                         ("collatz(937)", [937, 2812, 1406, 703, 2110, 1055, 3166, 1583, 4750, 2375, 7126, 3563, 10690, 5345, 16036, 8018, 4009, 12028, 6014, 3007, 9022, 4511, 13534, 6767, 20302, 10151, 30454, 15227, 45682, 22841, 68524, 34262, 17131, 51394, 25697, 77092, 38546, 19273, 57820, 28910, 14455, 43366, 21683, 65050, 32525, 97576, 48788, 24394, 12197, 36592, 18296, 9148, 4574, 2287, 6862, 3431, 10294, 5147, 15442, 7721, 23164, 11582, 5791, 17374, 8687, 26062, 13031, 39094, 19547, 58642, 29321, 87964, 43982, 21991, 65974, 32987, 98962, 49481, 148444, 74222, 37111, 111334, 55667, 167002, 83501, 250504, 125252, 62626, 31313, 93940, 46970, 23485, 70456, 35228, 17614, 8807, 26422, 13211, 39634, 19817, 59452, 29726, 14863, 44590, 22295, 66886, 33443, 100330, 50165, 150496, 75248, 37624, 18812, 9406, 4703, 14110, 7055, 21166, 10583, 31750, 15875, 47626, 23813, 71440, 35720, 17860, 8930, 4465, 13396, 6698, 3349, 10048, 5024, 2512, 1256, 628, 314, 157, 472, 236, 118, 59, 178, 89, 268, 134, 67, 202, 101, 304, 152, 76, 
            38, 19, 58, 29, 88, 44, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1], {'should_stop': True}),
                         ("collatz(1125899906842624)", [1125899906842624, 562949953421312, 281474976710656, 140737488355328, 70368744177664, 35184372088832, 17592186044416, 8796093022208, 4398046511104, 2199023255552, 1099511627776, 549755813888, 274877906944, 137438953472, 68719476736, 34359738368, 17179869184, 8589934592, 4294967296, 2147483648, 1073741824, 536870912, 268435456, 134217728, 67108864, 33554432, 16777216, 8388608, 4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1], {'should_stop': True})]
            for example, correct, options in testCases:
                if not Check.generator(example, correct, **options):
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
