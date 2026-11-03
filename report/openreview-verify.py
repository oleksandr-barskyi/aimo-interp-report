import sys, time
from playwright.sync_api import sync_playwright

EMAIL = "oleksandr.barskyi@karazin.ua"

if len(sys.argv) < 2:
    sys.exit("usage: openreview-verify.py <code>|--check [--register] [--out=dir]")
code = sys.argv[1]
register = "--register" in sys.argv
out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), ".")

with sync_playwright() as p:
    b = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
    pages = [pg for ctx in b.contexts for pg in ctx.pages if "/profile/activate" in pg.url]
    if not pages:
        sys.exit("no activation tab: open the newest 'OpenReview signup confirmation' link from gmail, add " + EMAIL + " in Emails and press Confirm (sends a new code)")
    page = pages[0]
    page.bring_to_front()
    page.locator("text=Emails").first.click()
    time.sleep(2)
    row = page.locator("input:visible").evaluate_all("(els, e) => els.findIndex(x => x.value === e)", EMAIL)
    if row < 0:
        sys.exit(EMAIL + " is not in the form any more: press Confirm again, a new code goes to karazin.ua")
    if code == "--check":
        sys.exit("ready: " + EMAIL + " waits for a code")
    page.locator("input[placeholder='Enter Verification Token']").fill(code)
    page.locator("button:visible", has_text="Verify").click()
    time.sleep(4)
    page.screenshot(path=out + "/or-verify.png")
    body = page.inner_text("body")
    ok = "has been verified" in body or "(Confirmed)" in body.split(EMAIL)[-1][:40]
    print("verified" if ok else "not verified, see or-verify.png")
    if not ok:
        sys.exit(1)
    if register:
        page.locator("text=Expertise").first.click()
        time.sleep(2)
        page.locator("button:visible", has_text="Register for OpenReview").click()
        time.sleep(8)
        page.screenshot(path=out + "/or-register.png")
        print(page.url)
        print(page.inner_text("body")[:800])
