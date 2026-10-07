import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
const root=path.resolve(process.env.OUTPUT_DIR||'output/Decision618_Deck_20261007');
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1920,height:1080}});
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
await page.goto(pathToFileURL(path.join(root,'Presentation_EN.html')).href);
await page.evaluate(()=>document.fonts.ready);
await page.addStyleTag({content:'.reveal{transition:none!important;opacity:1!important;transform:none!important}.deck-controls,.edit-hotzone,.edit-toggle,.notes-overlay{visibility:hidden!important}'});
const checks=[];
for(let i=0;i<7;i++){
 await page.evaluate(i=>window.deck.showSlide(i),i);
 checks.push(await page.evaluate(()=>{
  const s=document.querySelector('.slide.active'); const r=s.getBoundingClientRect();
  const nodes=[...s.querySelectorAll('[data-edit]')];
  const outside=nodes.filter(e=>{const b=e.getBoundingClientRect();return b.right>r.right+1||b.bottom>r.bottom+1||b.left<r.left-1||b.top<r.top-1;}).map(e=>e.textContent.trim());
  const overflow=nodes.filter(e=>e.scrollWidth>e.clientWidth+2||e.scrollHeight>e.clientHeight+2).map(e=>({text:e.textContent.trim(),scrollHeight:e.scrollHeight,clientHeight:e.clientHeight}));
  return {slide:window.deck.index+1,activeCount:document.querySelectorAll('.slide.active').length,outside,overflow,stageRatio:r.width/r.height};
 }));
 await page.screenshot({path:`tmp/d3_20261007/html-slide-${i+1}.png`});
}
await page.keyboard.press('Home');await page.keyboard.press('ArrowRight');
if(await page.evaluate(()=>deck.index)!==1)throw Error('Keyboard navigation failed');
await page.keyboard.press('e');if(!await page.evaluate(()=>deck.editing))throw Error('Edit mode failed');await page.keyboard.press('e');
await page.keyboard.press('n');if(!await page.evaluate(()=>document.querySelector('.notes-overlay').classList.contains('open')))throw Error('Notes failed');await page.keyboard.press('Escape');
await page.keyboard.press('t');if(!await page.evaluate(()=>deck.started))throw Error('Timer failed');await page.keyboard.press('t');
for(const viewport of [{width:1280,height:720},{width:390,height:844}]){
 await page.setViewportSize(viewport);await page.evaluate(()=>deck.showSlide(4));
 const ratio=await page.locator('.slide.active').boundingBox();if(Math.abs(ratio.width/ratio.height-16/9)>1e-5)throw Error('Stage ratio changed');
 await page.screenshot({path:`tmp/d3_20261007/html-${viewport.width}.png`});
}
await page.setViewportSize({width:1920,height:1080});
await page.pdf({path:path.join(root,'Presentation_EN.pdf'),width:'1920px',height:'1080px',printBackground:true,preferCSSPageSize:true});
await browser.close();
await fs.writeFile('tmp/d3_20261007/html-validation.json',JSON.stringify({checks,errors,keyboard:true,editing:true,notes:true,timer:true,viewports:['1920x1080','1280x720','390x844']},null,2));
console.log(JSON.stringify({checks,errors}));
if(errors.length||checks.some(c=>c.outside.length||c.overflow.length))process.exitCode=1;
