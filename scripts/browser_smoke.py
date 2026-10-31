"""Run against a locally started app; never submits individual records."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

base=os.getenv('HMDA_TEST_URL','http://127.0.0.1:8000/')
output=Path('outputs/screenshots');output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1,reduced_motion='reduce')
    titles={'overview':'Executive overview','lending':'Lending analytics','model':'Model intelligence','fairness':'Fair-lending monitoring','lineage':'Data lineage & governance'}
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    for route in ['overview','lending','model','fairness','lineage']:
        page.goto(base+'#'+route)
        page.wait_for_load_state('networkidle')
        expect(page.locator('h1')).to_have_text(titles[route])
        expect(page.locator('[role=status]')).to_have_count(0)
        assert page.locator('[role=alert]').count()==0, page.locator('body').inner_text()
        page.screenshot(path=str(output/f'{route}.png'),full_page=True)
        print('Rendered',route)
    page.goto(base+'#lending');page.wait_for_load_state('networkidle')
    page.get_by_label('year',exact=True).select_option('2025')
    page.wait_for_load_state('networkidle')
    expect(page.locator('main')).to_contain_text('25,000')
    page.goto(base+'#fairness');page.wait_for_load_state('networkidle')
    page.get_by_label('Minimum group size').select_option('0')
    page.wait_for_load_state('networkidle')
    expect(page.locator('main')).to_contain_text('Free Form Text Only')
    for route in ['overview','lending','model','fairness','lineage']:
        page.set_viewport_size({'width':390,'height':844})
        page.goto(base+'#'+route);page.wait_for_load_state('networkidle')
        expect(page.locator('h1')).to_have_text(titles[route])
        expect(page.locator('[role=status]')).to_have_count(0)
        expect(page.locator('.sidebar')).not_to_have_class('sidebar open')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),route
        page.screenshot(path=str(output/f'{route}-mobile.png'),full_page=True)
    page.get_by_role('button',name='Open navigation').click()
    page.get_by_role('link',name='Executive overview',exact=True).click()
    expect(page.locator('h1')).to_have_text('Executive overview')
    # Simulate a prefix-stripping reverse proxy while keeping the browser URL prefixed.
    prefix=base+'preview/demo/'
    requests=[]
    def proxy(route):
        target=route.request.url.replace(prefix,base,1)
        requests.append(route.request.url)
        response=route.fetch(url=target)
        route.fulfill(response=response)
    page.route(prefix+'**',proxy)
    page.goto(prefix+'#overview');page.wait_for_load_state('networkidle')
    expect(page.locator('main')).to_contain_text('125,000')
    assert any('/preview/demo/api/overview' in u for u in requests)
    assert any('/preview/demo/assets/' in u for u in requests)
    assert not errors,errors
    browser.close()
print('All five pages, filters and mobile navigation passed; no JavaScript errors.')
