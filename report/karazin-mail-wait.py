import os, re, subprocess, sys, time
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

EMAIL = "oleksandr.barskyi@karazin.ua"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
TASK = sys.argv[2] if len(sys.argv) > 2 else ""
ROUND_S = 600
ROUNDS = 4
os.makedirs(OUT, exist_ok=True)


def status(s):
    with open(os.path.join(OUT, "status.txt"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    print(s, flush=True)


def tg(pg, name, text):
    pg.bring_to_front()
    path = os.path.join(OUT, name)
    pg.screenshot(path=path)
    cmd = ["q", "send", path, "-m", text] + (["-t", TASK] if TASK else [])
    subprocess.run(cmd, check=False)


def host(pg):
    return urlparse(pg.url).netloc


def find(b):
    for c in b.contexts:
        for pg in c.pages:
            if host(pg) == "accounts.google.com":
                return pg
            if host(pg) == "mail.google.com" and EMAIL in pg.title():
                return pg
    return None


def skip_interstitials(pg):
    for t in ["Not now", "Skip", "Remind me later", "Cancel", "Done"]:
        loc = pg.get_by_role("button", name=t)
        if loc.count():
            loc.first.click()
            time.sleep(4)
            return True
    return False


with sync_playwright() as p:
    b = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
    pg = find(b)
    if pg is None:
        status("no google tab")
        sys.exit(1)
    sent = None
    started = time.time()
    resends = 0
    while True:
        url = pg.url
        if host(pg) == "mail.google.com":
            break
        if "challenge/dp" in url:
            txt = pg.inner_text("body")
            m = re.search(r"tap (\d+) on your phone", txt)
            num = m.group(1) if m else "?"
            if num != sent:
                sent = num
                status("prompt " + num)
                tg(pg, "prompt.png", f"Google просить підтвердити вхід у {EMAIL}: на iPhone в застосунку Gmail натисни «Так» і число {num}. Чекаю до {ROUND_S // 60} хв, потім надішлю новий запит (усього {ROUNDS}).")
            if time.time() - started > ROUND_S:
                resends += 1
                if resends >= ROUNDS:
                    status("gave up")
                    tg(pg, "gave-up.png", f"Вхід у {EMAIL} так і не підтвердили, сервіс зупинився. Відповідай на питання задачі {TASK}, коли будеш біля телефона.")
                    sys.exit(2)
                pg.bring_to_front()
                pg.get_by_role("button", name="Resend it").first.click()
                started = time.time()
                time.sleep(6)
                continue
        elif "rejected" in url or "signin/v2/deniedsigninrejected" in url:
            status("rejected " + url)
            tg(pg, "rejected.png", f"Google відхилив вхід у {EMAIL} з сервера: {pg.inner_text('body')[:300]}")
            sys.exit(3)
        elif "/signin/" in url and time.time() - started > ROUND_S:
            status("stuck " + url)
            sys.exit(5)
        elif host(pg) != "accounts.google.com" or "/signin/" not in url:
            if not skip_interstitials(pg):
                pg.goto("https://mail.google.com/mail/u/0/")
                time.sleep(8)
            continue
        time.sleep(10)

    status("signed in")
    tg(pg, "signed-in.png", f"Вхід у {EMAIL} підтверджено, сесія тримається в Chromium на CDP 127.0.0.1:9222.")
