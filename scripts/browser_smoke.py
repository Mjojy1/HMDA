"""Run against a locally started app; never submits individual records."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

base=os.getenv('HMDA_TEST_URL','http://127.0.0.1:8000/')
output=Path(os.getenv('HMDA_SCREENSHOT_DIR','outputs/screenshots'));output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1,reduced_motion='reduce')
    titles={'analyst':'Ask the analyst','overview':'Understand the lending picture','model':'Explore the model classifier','fairness':'Review group outcomes'}
    errors=[]
    failed_responses=[]
    if os.getenv('HMDA_TEST_DOMINO_AUTH')=='1':
        import requests
        response=requests.get('http://localhost:8899/access-token',timeout=10)
        response.raise_for_status()
        try: token=response.json()
        except ValueError: token=response.text
        if isinstance(token,dict):token=token.get('access_token',token.get('token'))
        def authenticate(route):
            route.continue_(headers={**route.request.headers,'Authorization':'Bearer '+token.strip()})
        page.route(base+'**',authenticate)
    page.on('console',lambda msg: errors.append(msg.text) if msg.type=='error' else None)
    page.on('response',lambda response: failed_responses.append((response.url,response.status)) if response.status>=400 else None)
    page.on('pageerror',lambda e: errors.append(str(e)))
    for route in ['overview','model','fairness','analyst']:
        page.goto(base+'#'+route)
        page.wait_for_load_state('networkidle')
        expect(page.locator('h1')).to_have_text(titles[route])
        expect(page.locator('[role=status]')).to_have_count(0)
        assert page.locator('[role=alert]').count()==0, page.locator('body').inner_text()
        page.screenshot(path=str(output/f'{route}.png'),full_page=True)
        print('Rendered',route)
    page.goto(base+'#model');page.wait_for_load_state('networkidle')
    expect(page.get_by_label('Contextual analyst')).to_have_count(0)
    page.get_by_label('These are fictional inputs for a test demonstration.').check()
    page.get_by_role('button',name='Classify test scenario').click()
    expect(page.get_by_text('Predicted historical class:',exact=False)).to_be_visible()
    expect(page.locator('main')).to_contain_text('Uncalibrated score')
    page.get_by_role('tab',name='Evaluation details').click()
    expect(page.locator('main')).to_contain_text('confusion matrix')
    page.get_by_role('tab',name='Methodology',exact=True).click()
    expect(page.locator('main')).to_contain_text('2021')
    page.get_by_role('tab',name='Summary',exact=True).click()
    page.get_by_role('button',name='Ask analyst',exact=True).click()
    expect(page.get_by_label('Contextual analyst')).to_be_visible()
    page.keyboard.press('Escape')
    expect(page.get_by_label('Contextual analyst')).to_have_count(0)
    page.goto(base+'#analyst');page.wait_for_load_state('networkidle')
    page.goto(base+'#overview');page.wait_for_load_state('networkidle')
    expect(page.locator('.primary-nav a')).to_have_count(4)
    expect(page.locator('.scroll-chart')).to_have_count(2)
    expect(page.get_by_label('loan_type',exact=True).locator('option[value="1"]')).to_have_text('Conventional')
    page.get_by_label('Predictor 2',exact=True).select_option('loan_type')
    expect(page.locator('.scroll-chart').last).to_contain_text('Conventional')
    page.get_by_label('year',exact=True).select_option('2025')
    page.wait_for_load_state('networkidle')
    expect(page.locator('main')).to_contain_text('25,000')
    page.goto(base+'#fairness');page.wait_for_load_state('networkidle')
    page.get_by_label('Minimum group size').select_option('0')
    page.wait_for_load_state('networkidle')
    expect(page.locator('main')).to_contain_text('Free Form Text Only')
    expect(page.get_by_role('img',name='Historical approval rate, Model-predicted approval rate',exact=True)).to_be_visible()
    expect(page.locator('table td').filter(has_text='Joint')).to_have_count(0)
    groups=page.locator('tbody tr td:first-child').all_text_contents()
    missing={'Missing','Race Not Available','Sex Not Available','Ethnicity Not Available','Free Form Text Only'}
    flags=[g in missing for g in groups]
    assert flags==sorted(flags),groups
    expect(page.locator('.recharts-bar')).to_have_count(2)
    expect(page.locator('main')).to_contain_text('Joint category is excluded')
    for route in ['overview','model','fairness','analyst']:
        page.set_viewport_size({'width':390,'height':844})
        page.goto(base+'#'+route);page.wait_for_load_state('networkidle')
        expect(page.locator('h1')).to_have_text(titles[route])
        expect(page.locator('[role=status]')).to_have_count(0)
        expect(page.locator('.primary-nav')).not_to_have_class('primary-nav open')
        if page.get_by_role('button',name='Close analyst',exact=True).count():
            page.get_by_role('button',name='Close analyst',exact=True).click()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),route
        page.screenshot(path=str(output/f'{route}-mobile.png'),full_page=True)
    page.get_by_role('button',name='Open navigation').click()
    page.get_by_role('link',name='Overview',exact=True).click()
    expect(page.locator('h1')).to_have_text('Understand the lending picture')
    if os.getenv('HMDA_TEST_DOMINO_AUTH')!='1':
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
    assert not failed_responses,failed_responses
    assert not errors,errors
    browser.close()
print('All four pages, predictor selection, loan labels, fairness ordering, filters and mobile navigation passed; no JavaScript errors.')
