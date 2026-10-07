import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
import {GlobalFonts} from '@napi-rs/canvas';

const ROOT=path.resolve(process.env.WORKSPACE_ROOT||'.');
const OUT=path.resolve(process.env.OUTPUT_DIR||path.join(ROOT,'output/Decision618_Deck_20261007_NextItNet'));
const BUILD=path.resolve(process.env.BUILD_DIR||path.join(ROOT,'tmp/d3_20261007_nextitnet'));
const SKILL=process.env.PRESENTATIONS_SKILL_DIR||'/Users/cuijiaming/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const PY=process.env.RUNTIME_PYTHON||'/Users/cuijiaming/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
process.env.RUNTIME_NODE_MODULES ||= '/Users/cuijiaming/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const SOURCE_DIR=process.env.SOURCE_DECK_DIR||path.join(ROOT,'presentations/Decision618_Deck_20261007');
const SOURCE=path.join(SOURCE_DIR,'Presentation_EN_20261007_v4.pptx');
const {finalizePresentation,applyPresentationChartFont}=await import(path.join(SKILL,'container_tools/artifact_tool_utils.mjs'));
const artifactRequire=createRequire(import.meta.resolve('@oai/artifact-tool'));
const {FontLibrary}=artifactRequire('skia-canvas');
const display='Bebas Neue',body='DM Sans';
for(const [family,file] of [[display,'BebasNeue.ttf'],[body,'DMSans.ttf']]){
 const font=path.join(SOURCE_DIR,'fonts',file);
 GlobalFonts.registerFromPath(font,family);FontLibrary.use(family,font);
}
await fs.mkdir(BUILD,{recursive:true});
const P=JSON.parse(await fs.readFile(path.join(OUT,'authoring/content.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
await fs.writeFile(path.join(BUILD,'source-inspection.ndjson'),(await p.inspect({kind:'slide,textbox,shape,chart,layout',maxChars:50000})).ndjson);
// Reuse the current nine-slide deck and its native framing.
if(p.slides.items.length!==9)throw Error('Source must have nine slides');
if(p.slides.items.length!==P.slides.length)throw Error('Content and slide counts differ');
const C={dark:'#171714',paper:'#F1E8D5',accent:'#E46242',ink:'#24231e',mutedDark:'#BDB8AD',mutedLight:'#656258'};
const html=[];
const esc=v=>String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const pos=(x,y,w,h)=>`left:${x*1.5}px;top:${y*1.5}px;width:${w*1.5}px;height:${h*1.5}px`;
let frag=[];
function ht(copy,x,y,w,h,size=28,{bold=false,color=C.paper,font=body}={}){
 const br=esc(copy).replaceAll('\n','<br>');
 frag.push(`<p data-edit style="position:absolute;${pos(x,y,w,h)};font-family:'${font}';font-size:${size*1.5}px;line-height:1.22;font-weight:${bold?600:400};color:${color}">${br}</p>`);
}
function text(s,copy,x,y,w,h,size=28,{bold=false,color=s.ink,font=body}={}){
 const shape=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 shape.text=copy;shape.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};
 ht(copy,x,y,w,h,size,{bold,color,font});return shape;
}
function foot(s,copy){text(s,copy,75,665,1080,39,16,{color:s.muted});}
function clearBody(s,i){
 // Retain the source slide, its master/layout, background, title and framing.
 const keep=new Set(s.shapes.items.slice(0,6).map(x=>x.id));
 for(const shape of [...s.shapes.items])if(!keep.has(shape.id))s.shapes.deleteById(shape.id);
 for(const chart of [...s.charts.items])s.charts.deleteById(chart.id);
 for(const t of [...s.tables.items])s.tables.deleteById(t.id);
 s.shapes.items[4].text=P.slides[i].title;
 s.shapes.items[4].text.style={typeface:display,fontSize:64,color:s.ink,autoFit:'none'};
}
function table(s,values,x,y,widths,heights,{size=25,highlightRow=-1}={}){
 const w=widths.reduce((a,b)=>a+b,0),h=heights.reduce((a,b)=>a+b,0);
 const t=s.tables.add({rows:values.length,columns:widths.length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.styleOptions={headerRow:false,bandedRows:false};
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:widths.length}).assign({fill:s.bg,textStyle:{typeface:body,fontSize:size,color:s.ink},margins:{left:12,right:12,top:10,bottom:10},anchor:'center'});
 t.borders.assign({style:'solid',fill:s.light?'#d2c8b7':'#555149',width:0.7});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=heights[r];
  if(r===0||r===highlightRow){
   const range=t.cells.block({row:r,column:0,rowCount:1,columnCount:widths.length});
   range.textStyle.bold=true;range.textStyle.color=r===highlightRow?C.accent:s.muted;
  }
 }
 frag.push(`<table class="evidence-table" style="position:absolute;${pos(x,y,w,h)};font-size:${size*1.5}px;color:${s.ink};border-color:${s.light?'#d2c8b7':'#555149'}"><colgroup>${widths.map(v=>`<col style="width:${v*1.5}px">`).join('')}</colgroup>${values.map((row,r)=>`<tr style="height:${heights[r]*1.5}px;color:${r===highlightRow?C.accent:r===0?s.muted:s.ink};font-weight:${r===0||r===highlightRow?600:400}">${row.map(v=>`<td data-edit>${esc(v).replaceAll('\n','<br>')}</td>`).join('')}</tr>`).join('')}</table>`);
 return t;
}
const originalHtml=await fs.readFile(path.join(SOURCE_DIR,'Presentation_EN.html'),'utf8');
const originalSections=[...originalHtml.matchAll(/<section class="slide[\s\S]*?<\/section>/g)].map(m=>m[0]);
for(let i=0;i<P.slides.length;i++){
 const s=p.slides.items[i];s.light=[1,2,4,6,8].includes(i);s.ink=s.light?C.ink:C.paper;s.muted=s.light?C.mutedLight:C.mutedDark;s.bg=s.light?C.paper:C.dark;
 // Existing header and numbering remain in their original positions.
 s.shapes.items[2].text=P.slides[i].label.toUpperCase();
 if(i>0)s.shapes.items[5].text=String(i+1).padStart(2,'0');
 frag=[`<div class="rule"></div><div class="topline"><span>DECISION 618 / Movie recommendations</span><span>${esc(P.slides[i].label)}</span></div>`];
 if(i>0)ht(P.slides[i].title,75,115,1130,103,64,{color:s.ink,font:display});
 ht(String(i+1).padStart(2,'0'),1182,669,50,29,17,{color:s.muted});
 if(i===0){
  s.shapes.items[1].text='DECISION 618 / MOVIE RECOMMENDATIONS';
  ht('MOVIE',75,152,1080,166,130,{font:display});
  ht('RECOMMENDATIONS',75,313,1090,170,130,{font:display,color:C.accent});
  ht('Course methods, timestamps and adapted NextItNet',77,475,1095,62,28,{color:s.muted});
  ht('Jiaming Cui, Joonse Lim, Tung Jerateepkulmeth',77,617,1100,47,19,{color:s.muted});
  ht('MovieLens 1M / October 7, 2026',77,551,1080,45,22,{color:s.muted});
 }else if(i===1){
  // Import loses the workbook relationship. Rebuild this native chart from the
  // audited literal data and original chart styling, then embed its workbook.
  for(const old of [...s.charts.items])s.charts.deleteById(old.id);
  const ch=s.charts.add('bar',{position:{left:570,top:300,width:640,height:290},categories:['Same timestamp','1 to 60 seconds','More than 60 seconds'],
   series:[{name:'Rate',values:[P.eda.equal_second_fraction,P.eda.within_60_seconds_fraction-P.eda.equal_second_fraction,1-P.eda.within_60_seconds_fraction].map(v=>Number(v.toFixed(8))),fill:C.accent,valuesFormatCode:'0.0%'}],
   hasLegend:false,barOptions:{direction:'bar',grouping:'clustered',gapWidth:90},
   xAxis:{textStyle:{typeface:body,fontSize:21,fill:s.ink},line:{fill:s.muted,width:1}},
   yAxis:{min:0,max:.65,numberFormatCode:'0.0%',textStyle:{typeface:body,fontSize:18,fill:s.muted},majorGridlines:{fill:'#dfd5c3',width:.5}},
   dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:body,fontSize:23,bold:true,fill:s.ink}},chartFill:s.bg,plotAreaFill:s.bg});
  applyPresentationChartFont(ch,{fontFamily:body});
  html.push(originalSections[i].replace(/data-seconds="\d+"/,`data-seconds="${P.slides[i].seconds}"`));
 }else{
  clearBody(s,i);
  if(i===2){
   text(s,'CLASSROOM BENCHMARK',75,233,560,72,44,{font:display});
   text(s,'Exact 99/1 split, seed 144\n20-fold CV selects rank 8',75,318,555,96,27);
   text(s,'FRESH REPLICATION',720,233,490,72,44,{font:display,color:C.accent});
   text(s,'400 R fits: 20 ranks × 20 folds\nFresh CV selects rank 9',720,318,480,96,27);
   table(s,[['Rank-8 test','RMSE','MAE','OSR²'],['Instructor notebook','0.919','0.709','0.332'],['Our classroom refit','0.9198','0.7106','0.3300']],75,456,[485,215,215,215],[48,55,55],{size:24,highlightRow:2});
   foot(s,'Rank 8 retained for the classroom comparison. Original R seed/version are unrecorded.');
  }else if(i===3){
   text(s,'Same 1-5 rating target. The split follows the prediction question.',75,228,1120,50,24,{color:s.muted});
   table(s,[['Partition','Random 99/1','Global time 80/10/10'],['Train','990,206','800,164'],['Validation','20-fold CV in train','100,024'],['Test','10,003','100,021']],75,300,[400,365,365],[58,58,58,58],{size:25});
   text(s,'Random: complete missing ratings.\nGlobal time: predict future ratings.',75,559,565,86,25);
   text(s,'Keep timestamp ties together.\nRefit train + validation before test.',720,559,488,86,25);
   foot(s,'Lecture 2: match the split to the use. Lecture 7: random rating completion.');
  }else if(i===4){
   text(s,'Ridge combines CF predictions and features. Lower RMSE is better.',75,223,1120,45,23,{color:s.muted});
   const labels=['Training mean','Collaborative filtering','CF + metadata','CF + metadata + timestamp'];
   const values=[['Model','Random RMSE','Future RMSE'],...labels.map((name,j)=>[name,...P.rmse[j].map(v=>v.toFixed(4))])];
   table(s,values,75,278,[560,285,285],[58,58,58,58,58],{size:26,highlightRow:4});
   text(s,'Timestamp increment vs. metadata: 0.0050 random / 0.0302 future',75,591,1125,52,27);
   foot(s,'RMSE in rating points. Same holdout within each column. Columns cover different populations.');
  }else if(i===5){
   text(s,'UTC calendar, elapsed days, and counts/gaps from earlier ratings.\nFrozen fitted history. Equal-second entries excluded.',75,235,1130,98,27);
   const ci=g=>`[${g.ci95.map(v=>v.toFixed(4)).join(', ')}]`;
   table(s,[['Timestamp vs. metadata','RMSE reduction','Paired 95% interval'],['Random',P.gains[0].improvement_RMSE.toFixed(4),ci(P.gains[0])],['Future',P.gains[1].improvement_RMSE.toFixed(4),ci(P.gains[1])]],75,371,[455,275,400],[58,62,62],{size:25});
   text(s,'Warm ratings: 23.84% in validation / 95.70% in final test.\nThis cohort change limits transfer of selected parameters.',75,579,1130,80,25,{color:s.muted});
   foot(s,'Intervals condition on this split and selection. Future CF is worse than training mean.');
  }else if(i===6){
   text(s,'Full sample: 6,038 users. Strictly later test rating: 3,494 users.',75,223,1130,47,24,{color:s.muted});
   const values=[['Model','Full sample\nHR@10','Full sample\nNDCG@10','Later test\nHR@10','Later test\nNDCG@10'],
    ...P.nextitnet.ranking.map(r=>[r.model==='NextItNet'?'Adapted NextItNet':r.model,
     (100*r['HR@10']).toFixed(2)+'%',r['NDCG@10'].toFixed(4),
     (100*r['strict_HR@10']).toFixed(2)+'%',r['strict_NDCG@10'].toFixed(4)])];
   table(s,values,75,282,[340,190,200,200,200],[74,58,58,58,58],{size:24,highlightRow:4});
   text(s,'NextItNet vs. item-kNN: +12.54 percentage points in full-sample HR@10',75,611,1130,43,25);
   foot(s,'Saved experiment, September 12 UTC. Later-test subset removes boundary ties, not global future data.');
  }else if(i===7){
   text(s,'Separate objectives and holdouts require separate comparisons.',75,228,1130,48,25,{color:s.muted});
   table(s,[['','Rating prediction','Next-item ranking'],
    ['Target','1-5 star rating','Next rated movie'],
    ['Holdout','Random 99/1 or\nglobal time 80/10/10','Last test / previous validation\nfor each user'],
    ['Metric','RMSE: lower is better','HR@10, NDCG@10: higher']],75,292,[220,430,480],[58,64,105,70],{size:24});
   text(s,'Per-user holdouts can include other users\' later ratings.\nSVD here optimizes stars. A shared global-time ranking test remains.',75,611,1100,79,23,{color:s.muted});
  }else{
   text(s,'MOVIE',75,240,500,115,84,{font:display,color:C.accent});
   text(s,'STARTS',75,343,500,115,84,{font:display,color:C.accent});
   text(s,'Proposed pilot measure:\nrecommendation-led starts\nper exposed user.',75,462,555,120,27);
   for(const [y,title,copy] of [[241,'Before a pilot','Compare rankings on shared time windows.'],[365,'Candidate systems','Timestamp ensemble and adapted\nNextItNet, with cold-user fallbacks.'],[489,'Guardrails','Completion, catalog concentration,\nlatency and privacy.']]){
    text(s,title,707,y,490,45,24,{bold:true});text(s,copy,707,y+46,490,72,24);
   }
   text(s,'Repeat evaluation across time windows before a limited randomized pilot.\nMovieLens has no exposure, revenue or cost data to establish ROI.',75,611,1115,51,20,{color:s.muted});
  }
 }
 if(i!==1)html.push(`<section class="slide ${s.light?'light ':''}${i===0?'active visible':''}" data-seconds="${P.slides[i].seconds}">${frag.join('')}</section>`);
 s.speakerNotes.textFrame.setText(P.slides[i].script+'\n\n中文讲解\n'+P.slides[i].chinese+'\n\nPlanned time: '+P.slides[i].seconds+' seconds\n\nSources\n'+P.slides[i].sources.join('\n')+'\nCourse page references use PDF positions. Exact Canvas file IDs/hashes: course_rebuild_20261007/sources/course_manifest.json.\nMovieLens: https://grouplens.org/datasets/movielens/1m/');
}
let web=originalHtml.replace(/<title>[\s\S]*?<\/title>/,'<title>Decision 618: Rating prediction and NextItNet, October 7</title>');
web=web.replace(/<main class="deck-stage" id="deckStage">[\s\S]*?<\/main>/,`<main class="deck-stage" id="deckStage">${html.join('')}</main>`);
web=web.replace(/const SLIDE_NOTES=[\s\S]*?;\nclass SlidePresentation/,`const SLIDE_NOTES=${JSON.stringify(P.slides)};\nclass SlidePresentation`);
web=web.replaceAll('decision618-d3-cinema-edits','decision618-oct7-rating-nextitnet-v4-edits');
web=web.replaceAll('Decision618_Presentation_EN.html','Decision618_Presentation_EN_20261007.html');
web=web.replace('Sources: GroupLens MovieLens 1M; Yuan et al. (2019); Recommenders 1.2.1; DECISION 618 lecture 7; local experiment summary and per-user metrics.','Sources: ${s.sources.join("; ")}');
web=web.replace('</style>','.evidence-table{table-layout:fixed;border-collapse:collapse;font-family:"DM Sans";line-height:1.2}.evidence-table td{border:1px solid;border-color:inherit;padding:15px 18px;vertical-align:middle;text-align:left}\n</style>');
await fs.writeFile(path.join(OUT,'Presentation_EN.html'),web);
const candidate=path.join(BUILD,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const final=path.join(OUT,'Presentation_EN_20261007_v4.pptx');
const sourceSha=crypto.createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:PY,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...['3','4','5','6','7','8'].flatMap(n=>['--require-native-table-slide',n])],
 explicitTotalSlideCount:P.slides.length,requiredNativeChartOwnerSlides:[2],requiredNativeTableOwnerSlides:[3,4,5,6,7,8],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[display,body],referencePath:SOURCE,referenceSha256:sourceSha},
 verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'pptx-validation-v4.json')});
console.log('Finalized',JSON.stringify(result));
const verified=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<P.slides.length;i++){
 const slide=verified.slides.items[i];const png=await verified.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,`pptx-slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await slide.export({format:'layout'});await fs.writeFile(path.join(BUILD,`pptx-slide-${i+1}.layout.json`),await layout.text());
}
console.log('Rendered',P.slides.length,'updated slides');
