const $=id=>document.getElementById(id);

const NODE_NAMES={load_data:'加载数据',profile:'数据画像',goal:'解析实验目标',recipe_hit:'配方检索',plan:'生成实验方案',plan_fallback:'使用保底方案',plan_invalid:'方案校验',execute:'执行与评估',attribute:'失败原因分析',attribute_fallback:'归因回退规则',replan:'调整方案',export:'生成实验报告',error:'实验出错',done:'实验结束'};
const CAUSES={missing_values_poor_imputation:'缺失值填充不足',class_imbalance:'类别不均衡',metric_weight_mismatch:'权重与指标方向不匹配',feature_scale:'特征未标准化',overfit:'过拟合',underfit:'欠拟合',budget_exceeded:'超出时间预算',execution_failed:'执行失败',unknown_shortfall:'指标未达标'};
const SOURCE_NAMES={template:'模板规划',attribution:'归因重规划',recipe:'复用同任务配方',transfer:'跨任务配方迁移',reproduce:'按方案复现',llm:'大模型规划'};
const MODEL_NAMES={nearest_centroid:'最近质心',logistic_gd:'逻辑回归',linear_gd:'线性回归',random_forest:'随机森林'};
const OP_NAMES={impute_zero:'零填充',impute_mean:'均值填充',impute_median:'中位数填充',drop_missing_rows:'删除缺失行',scale_standard:'标准化',onehot:'独热编码',variance_filter:'低方差过滤',log1p:'log1p 变换'};

let tasks=[],active=null,generation=0,busy=false,selected=new Set();
let events=[],stream=null,lastSeq=0,previewTimer=null,previewToken=0,previewData=null,openSeqs=new Set();
let target={threshold:null,higher:true},metricName='';
const LOWER_METRICS=['mae','rmse'];

async function api(path,options){
  const r=await fetch(path,{...options,signal:AbortSignal.timeout(20000)});
  if(!r.ok){
    let detail='';
    try{const data=await r.json();detail=data&&data.detail?String(data.detail):''}catch(_){}
    throw new Error(detail||('请求失败（'+r.status+'）'));
  }
  return r.json();
}
function message(text,error=false){$('message').textContent=text;$('message').className=error?'error':''}
function num(v){return typeof v==='number'&&Number.isFinite(v)?v.toFixed(4):'—'}
function parseJson(text){
  try{return JSON.parse(text)}catch(_){
    // 兼容旧 trace：Python 的 json.dumps 默认会写出 NaN/Infinity，浏览器不认识
    try{return JSON.parse(text.replace(/-?\bNaN\b/g,'null').replace(/-?\bInfinity\b/g,'null'))}catch(__){return null}
  }
}
function el(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!=null)node.textContent=text;return node}
function updateMetricLabels(){
  const name=(metricName||'').toUpperCase();
  const lower=LOWER_METRICS.includes((metricName||'').toLowerCase());
  $('metriclabel').textContent=name?('当前指标 · '+name+'（'+(lower?'越低越好':'越高越好')+'）'):'当前指标';
  $('targetlabel').textContent=name?('目标阈值 · '+name):'目标阈值';
}

function closeStream(){if(stream){stream.close();stream=null}}

function addBadge(text,kind){
  const badge=el('span','badge'+(kind?' '+kind:''),text);
  $('planbadges').append(badge);
}

function normalizePlan(output){
  if(!output)return null;
  if(output.plan)return {plan:output.plan,rationale:output.rationale||null};
  return {plan:output,rationale:null};
}
function planFromEvents(){
  const planEvent=[...events].reverse().find(e=>e.node==='plan');
  if(!planEvent)return null;
  const hit=[...events].reverse().find(e=>e.node==='recipe_hit');
  return {...normalizePlan(planEvent.output),recipe:hit?hit.output:null};
}
function deriveSteps(plan){
  const steps=(plan.ops||[]).map(op=>({stage:'预处理',action:OP_NAMES[op.op]||op.op,reason:''}));
  const model=plan.model||{};
  steps.push({stage:'模型',action:MODEL_NAMES[model.kind]||model.kind||'—',reason:''});
  steps.push({stage:'评估',action:((plan.eval||{}).n_splits||5)+' 折交叉验证',reason:''});
  return steps;
}
function renderPlanCard(data){
  const box=$('plan');
  $('planbadges').replaceChildren();
  if(!data||!data.plan){
    box.replaceChildren(el('p','plan-empty','暂时没有方案草案：确认任务数据可用后会自动生成。'));
    return;
  }
  const plan=data.plan||{},recipe=data.recipe||{};
  addBadge(SOURCE_NAMES[plan.source]||plan.source||'模板规划',plan.source==='transfer'?'transfer':(plan.source==='recipe'?'exact':'quiet'));
  if(recipe.match==='similar'){
    const prefix=plan.source==='transfer'?'已迁移':'可迁移';
    addBadge(prefix+' · '+(recipe.source_task_id||'历史任务')+' · 相似度 '+Number(recipe.similarity||0).toFixed(2),'transfer');
  }else if(recipe.match==='exact'){
    addBadge('命中同任务配方','exact');
  }
  box.replaceChildren();
  const profile=data.profile;
  if(profile){
    box.append(el('p','plan-meta','画像：'+(profile.n_rows??'—')+' 行 · '+(profile.n_features??'—')+' 特征 · 缺失 '+(Number(profile.missing_ratio||0)*100).toFixed(2)+'%'));
  }
  const steps=(data.rationale&&data.rationale.length)?data.rationale:deriveSteps(plan);
  const list=document.createElement('ol');list.className='steps';
  for(const step of steps){
    const item=document.createElement('li');
    item.append(el('span','stage',step.stage||''));
    item.append(el('span','action',step.action||step.op||''));
    item.append(el('span','reason',step.reason||''));
    list.append(item);
  }
  box.append(list);
}

function renderGoalHint(goal){
  const applied=(goal&&goal.applied)||{};
  const text=$('goal').value.trim();
  if(!text){$('goalhint').textContent='可留空，直接使用任务默认阈值与预算';return}
  if(!Object.keys(applied).length){$('goalhint').textContent='未识别到数值约束，沿用任务默认配置';return}
  const parts=[];
  if(applied.threshold!=null)parts.push('阈值 '+applied.threshold);
  if(applied.budget_seconds!=null)parts.push('预算 '+applied.budget_seconds+' 秒');
  if(applied.max_rounds!=null)parts.push('最多 '+applied.max_rounds+' 轮');
  $('goalhint').textContent='已识别：'+parts.join(' · ');
}

function roundPoints(){
  return events.filter(e=>e.node==='execute').map((e,i)=>{
    const o=e.output||{};
    return {
      round:(e.input&&e.input.round)||i+1,
      mean:Number.isFinite(o.mean)?o.mean:null,
      passed:!!o.passed,
      holdout:Number.isFinite(o.holdout)?o.holdout:null,
    };
  });
}
function drawChart(points){
  const box=$('chart');
  const pts=points.filter(p=>typeof p.mean==='number');
  if(!pts.length){box.replaceChildren(el('p','muted','每轮实验结束后在这里画出指标走势'));return}
  const W=660,H=220,L=52,R=22,T=22,B=34;
  const values=pts.map(p=>p.mean);
  const threshold=target.threshold;
  const all=threshold==null?values.slice():values.concat([threshold]);
  let lo=Math.min(...all),hi=Math.max(...all);
  if(hi-lo<0.02){const mid=(hi+lo)/2;lo=mid-0.02;hi=mid+0.02}
  const padding=(hi-lo)*0.18;lo-=padding;hi+=padding;
  const x=i=>pts.length===1?(L+(W-L-R)/2):L+i*(W-L-R)/(pts.length-1);
  const y=v=>H-B-((v-lo)/(hi-lo))*(H-T-B);
  let grid='';
  for(let k=0;k<=2;k++){
    const v=lo+(hi-lo)*k/2,gy=y(v);
    grid+='<line x1="'+L+'" y1="'+gy.toFixed(1)+'" x2="'+(W-R)+'" y2="'+gy.toFixed(1)+'" stroke="#eef2f7"/>';
    grid+='<text x="'+(L-8)+'" y="'+(gy+4).toFixed(1)+'" text-anchor="end" font-size="11" fill="#8794a8">'+v.toFixed(3)+'</text>';
  }
  let thresholdLine='';
  if(threshold!=null&&threshold>=lo&&threshold<=hi){
    const ty=y(threshold);
    thresholdLine='<line x1="'+L+'" y1="'+ty.toFixed(1)+'" x2="'+(W-R)+'" y2="'+ty.toFixed(1)+'" stroke="#b3261e" stroke-dasharray="6 4" opacity=".7"/>';
    thresholdLine+='<text x="'+(W-R)+'" y="'+(ty-6).toFixed(1)+'" text-anchor="end" font-size="11" fill="#b3261e">阈值 '+threshold+'</text>';
  }
  const line=pts.map((p,i)=>(i?'L':'M')+x(i).toFixed(1)+','+y(p.mean).toFixed(1)).join(' ');
  const dots=pts.map((p,i)=>{
    const cy=y(p.mean),color=p.passed?'#0f7b5f':'#c47a00';
    return '<circle cx="'+x(i).toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="4.5" fill="'+color+'"/>'
      +'<text x="'+x(i).toFixed(1)+'" y="'+(cy-10).toFixed(1)+'" text-anchor="middle" font-size="11" fill="#33445c">'+p.mean.toFixed(3)+'</text>'
      +'<text x="'+x(i).toFixed(1)+'" y="'+(H-12)+'" text-anchor="middle" font-size="11" fill="#7b879b">第 '+p.round+' 轮</text>';
  }).join('');
  box.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="各轮实验指标曲线">'+grid+thresholdLine
    +'<path d="'+line+'" fill="none" stroke="#079da7" stroke-width="2.2" stroke-linejoin="round"/>'+dots+'</svg>';
  $('chartnote').textContent=(threshold==null?'':'目标 '+(metricName||'指标').toUpperCase()+' '+(target.higher?'≥ ':'≤ ')+threshold+' · ')+'共 '+pts.length+' 轮';
}

function renderMetrics(){
  const pts=roundPoints();
  if(pts.length){
    const last=pts[pts.length-1];
    $('score').textContent=num(last.mean);
    $('round').textContent=String(pts.length);
    $('holdout').textContent=last.holdout==null?'—':num(last.holdout);
  }else{
    $('score').textContent='—';$('round').textContent='—';$('holdout').textContent='—';
    $('chartnote').textContent=target.threshold==null?'':'目标 '+(metricName||'指标').toUpperCase()+' '+(target.higher?'≥ ':'≤ ')+target.threshold;
  }
  drawChart(pts);
}

function updateLinks(){
  const exp=[...events].reverse().find(e=>e.node==='export');
  if(!exp||!active)return;
  $('report').href='/api/runs/'+encodeURIComponent(active)+'/report';
  $('report').hidden=false;
  const out=exp.output||{};
  if(out.model_path||out.model_file){
    $('model').href='/api/runs/'+encodeURIComponent(active)+'/model';
    $('model').hidden=false;
  }
}

function renderTimeline(){
  const tl=$('timeline');
  tl.replaceChildren();
  if(!events.length){tl.append(el('p','muted','正在等待实验事件…'));return}
  for(const e of events){
    const o=e.output||{};
    const row=el('article','event');
    row.append(el('h3',null,String(e.seq).padStart(2,'0')+' · '+(NODE_NAMES[e.node]||e.node)));
    row.append(el('p','muted',(e.tool||'流程控制')+' · '+(e.duration_ms==null?'耗时未记录':Number(e.duration_ms).toFixed(0)+' ms')));
    let text='';
    if(e.node==='goal'){
      const applied=o.applied||{};
      text='识别约束：'+(Object.keys(applied).length?Object.entries(applied).map(([k,v])=>k+'='+v).join('，'):'无，沿用任务默认');
    }
    if(e.node==='recipe_hit'){
      text=o.match==='similar'
        ?'跨任务迁移：复用 '+(o.source_task_id||'历史任务')+' 的成功配方，画像相似度 '+Number(o.similarity||0).toFixed(2)
        :'命中同画像配方 '+String(o.recipe_id||'');
    }
    if(e.node==='plan_fallback')text='大模型方案不可用，已回退模板：'+(o.error||'原因未记录');
    if(e.node==='attribute_fallback')text='大模型归因不可用，已回退规则库：'+(o.error||'原因未记录');
    if(e.node==='plan'){
      const info=normalizePlan(o)||{},p=info.plan||{};
      text='来源：'+(SOURCE_NAMES[p.source]||p.source||'模板')+' · '+(p.ops||[]).length+' 个预处理算子 · 模型 '+(MODEL_NAMES[(p.model||{}).kind]||(p.model||{}).kind||'—');
    }
    if(e.node==='execute'){
      if(o.status&&o.status!=='ok'){
        text='执行失败（'+o.status+'）：'+(o.error||'沙箱未返回可用结果');
      }else{
        const name=(o.metric||metricName||'指标').toUpperCase();
        const lower=LOWER_METRICS.includes(String(o.metric||metricName||'').toLowerCase());
        const bound=(target.threshold==null||(o.metric&&metricName&&o.metric!==metricName))?'':(' '+(lower?'≤':'≥')+' '+target.threshold);
        text=name+' '+num(o.mean)+bound+' · '+(o.passed?'已达标':'未达标')+'（'+(lower?'越低越好':'越高越好')+'） · '+Number(o.seconds||0).toFixed(2)+' 秒';
        if(typeof o.holdout==='number')text+='（留出集 '+num(o.holdout)+'，不参与达标判定）';
      }
    }
    if(e.node==='attribute'){
      const src=o.source==='llm'?'大模型归因':o.source==='rules'?'规则库归因':'';
      const parts=(o.causes||[]).map(c=>(CAUSES[c.cause]||c.cause)+'：'+c.evidence).join('；');
      text=(src?src+'：':'')+(o.summary?o.summary+'；':'')+parts;
    }
    if(e.node==='error')text=o.error||'实验发生错误';
    if(e.node==='done'){
      text=o.passed?'实验完成，目标已达成':o.status==='error'?'实验出错，请查看详情':'实验完成，尚未达标';
      $('runstatus').textContent=text;
      if(o.error)text+='：'+o.error;
    }
    if(text){
      const p=el('p',e.node==='error'?'error':null,text);
      row.append(p);
    }
    const details=document.createElement('details');
    details.append(el('summary',null,'查看步骤数据'));
    details.append(el('pre',null,JSON.stringify({input:e.input,output:o},null,2)));
    if(openSeqs.has(e.seq))details.open=true;
    details.addEventListener('toggle',()=>{details.open?openSeqs.add(e.seq):openSeqs.delete(e.seq)});
    row.append(details);
    tl.append(row);
  }
  updateLinks();
}

function renderAll(){
  renderTimeline();
  renderMetrics();
  renderPlanCard(planFromEvents()||previewData);
}

async function previewPlan(){
  const taskId=$('task').value;
  if(!taskId)return;
  const token=++previewToken;
  try{
    const data=await api('/api/plan/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId,goal:$('goal').value})});
    if(token!==previewToken)return;
    previewData=data;
    metricName=data.task.metric||metricName;
    target={threshold:Number(data.task.threshold),higher:!!data.task.higher_is_better};
    $('target').textContent=(target.higher?'≥ ':'≤ ')+target.threshold;
    updateMetricLabels();
    renderGoalHint(data.goal);
    if(!active)renderPlanCard(data);
    renderMetrics();
  }catch(e){
    if(token!==previewToken)return;
    previewData=null;
    $('goalhint').textContent='';
    if(!active){
      $('planbadges').replaceChildren();
      $('plan').replaceChildren(el('p','plan-empty','方案预览失败：'+e.message));
    }
  }
}
function schedulePreview(){clearTimeout(previewTimer);previewTimer=setTimeout(previewPlan,450)}

function finishRun(reason){
  if(!busy)return;
  busy=false;
  $('start').disabled=false;
  const done=[...events].reverse().find(e=>e.node==='done');
  if(reason)$('runstatus').textContent=reason;
  else if(done){
    const o=done.output||{};
    $('runstatus').textContent=o.passed?'实验完成，目标已达成':(o.status==='error'?'实验出错，请查看轨迹':'实验完成，尚未达标');
  }
  history();loadStats();
}

function watchLive(id,taskId){
  closeStream();
  active=id;
  const token=++generation;
  events=[];lastSeq=0;openSeqs.clear();
  $('score').textContent='—';$('round').textContent='—';$('holdout').textContent='—';
  $('report').hidden=true;$('model').hidden=true;
  $('runstatus').textContent='实验运行中 · '+id;
  $('timeline').replaceChildren(el('p','muted','正在连接实时事件流…'));
  renderPlanCard(previewData);
  renderMetrics();
  const es=new EventSource('/api/runs/'+encodeURIComponent(id)+'/stream');
  stream=es;
  let retries=0,fallback=false,received=false,watchdog=0;
  // SSE 不是 200（例如 404）时浏览器会直接关闭连接且**不再重连**；
  // 所以不能只靠 onerror 的次数判断，必须主动回退到轮询。
  const startPolling=reason=>{
    if(fallback||token!==generation)return;
    fallback=true;clearTimeout(watchdog);
    try{es.close()}catch(_){}
    if(reason)message(reason);
    $('runstatus').textContent='实验运行中 · '+id+'（已切换轮询）';
    pollLive(id,token);
  };
  watchdog=setTimeout(()=>{
    if(!received)startPolling('实时事件流未在 5 秒内送达，已切换为轮询。');
  },5000);
  es.onopen=()=>{retries=0};
  es.onmessage=ev=>{
    if(token!==generation)return;
    const event=parseJson(ev.data);
    if(!event)return;
    received=true;clearTimeout(watchdog);
    if(typeof event.seq==='number'&&event.seq<=lastSeq)return;
    if(typeof event.seq==='number')lastSeq=event.seq;
    events.push(event);
    renderAll();
    if(event.node==='done'){closeStream();finishRun()}
  };
  es.onerror=()=>{
    if(token!==generation||fallback)return;
    if(es.readyState===2){startPolling('实时事件流不可用，已切换为轮询。');return}
    if(++retries>2)startPolling('实时事件流多次重连失败，已切换为轮询。');
  };
}
async function pollLive(id,token){
  const began=Date.now();
  while(token===generation){
    let list=null,summary=null;
    try{list=await api('/api/runs/'+encodeURIComponent(id)+'/trace')}catch(_){}
    try{summary=await api('/api/runs/'+encodeURIComponent(id))}catch(_){}
    if(token!==generation)return;
    if(list){
      events=list;lastSeq=list.length?list[list.length-1].seq:0;
      renderAll();
      if(list.some(e=>e.node==='done')){closeStream();finishRun();return}
    }
    if(summary&&summary.status&&summary.status!=='running'){
      // 后端已经结束但 trace 里没有 done（例如工作线程提前失败）：也要把界面收尾
      const reason=summary.error?('实验出错：'+summary.error):(summary.passed?'实验完成，目标已达成':'实验完成，尚未达标');
      closeStream();finishRun(reason);return;
    }
    if(Date.now()-began>600000){
      message('等待超过十分钟，已停止自动刷新。可在历史记录中重新查看。',true);
      finishRun('等待超时，已停止自动刷新。');
      return;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
}

function watchHistory(id){
  if(busy){message('请等待当前实验结束后再查看历史。');return}
  closeStream();
  active=id;generation++;events=[];openSeqs.clear();
  $('score').textContent='—';$('round').textContent='—';$('holdout').textContent='—';
  $('report').hidden=true;$('model').hidden=true;
  $('runstatus').textContent=id;
  $('timeline').replaceChildren(el('p','muted','正在加载实验轨迹…'));
  api('/api/runs/'+encodeURIComponent(id)+'/trace').then(list=>{
    if(active!==id)return;
    events=list;renderAll();
  }).catch(()=>message('轨迹读取失败，请稍后重试。',true));
}

function resetView(){
  closeStream();generation++;active=null;events=[];openSeqs.clear();busy=false;
  $('score').textContent='—';$('round').textContent='—';$('holdout').textContent='—';
  $('report').hidden=true;$('model').hidden=true;$('runstatus').textContent='';
  $('timeline').innerHTML='<div class="empty"><div class="rail">○<br>│<br>○<br>│<br>○</div><h3>选择任务，开始一次实验</h3><p>实验轨迹会通过 SSE 实时推送到这里</p></div>';
  $('start').disabled=!tasks.length;
  renderMetrics();
  renderPlanCard(previewData);
}

async function removeRun(runId,button){
  if(!confirm('确定删除实验记录 '+runId+' ？将同时删除其轨迹、报告与复现包文件。'))return;
  button.disabled=true;
  try{
    await api('/api/runs/'+encodeURIComponent(runId),{method:'DELETE'});
    message('已删除实验记录 '+runId+'。');
    if(active===runId)resetView();
    await history();loadStats();
  }catch(e){message('删除失败：'+e.message,true);button.disabled=false}
}
function updateBatch(){
  const n=selected.size,all=$('history').querySelectorAll('input.pick');
  $('batchdel').disabled=n===0;
  $('batchdel').textContent=n?'删除所选（'+n+'）':'删除所选';
  $('pickall').checked=all.length>0&&n===all.length;
  $('pickall').indeterminate=n>0&&n<all.length;
}
async function removeSelected(){
  const ids=[...selected];
  if(!ids.length)return;
  if(!confirm('确定删除选中的 '+ids.length+' 条实验记录？将同时删除其轨迹、报告与复现包文件（配方库保留）。'))return;
  $('batchdel').disabled=true;
  try{
    const r=await api('/api/runs',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_ids:ids})});
    const ok=(r.counts&&r.counts.deleted)||0,bad=(r.counts&&r.counts.failed)||0;
    let msg='已删除 '+ok+' 条实验记录。';
    if(bad)msg+=' '+bad+' 条未删除：'+(r.failed||[]).map(f=>f.run_id+'（'+f.reason+'）').join('；');
    message(msg,bad>0);
    if(active&&(r.deleted||[]).some(d=>d.run_id===active))resetView();
    selected.clear();
    await history();loadStats();
  }catch(e){message('批量删除失败：'+e.message,true)}
  finally{updateBatch()}
}

async function history(){
  try{
    const rows=await api('/api/runs');
    selected=new Set();
    const box=$('history');box.replaceChildren();
    if(!rows.length){box.append(el('p','muted','暂无实验记录'));return}
    for(const r of rows){
      const item=el('div','history-item');
      const pick=document.createElement('input');
      pick.type='checkbox';pick.className='pick';pick.title='选择这条记录';
      pick.setAttribute('aria-label','选择实验记录 '+r.run_id);pick.dataset.rid=r.run_id;
      pick.onchange=()=>{pick.checked?selected.add(r.run_id):selected.delete(r.run_id);updateBatch()};
      const button=el('button','history',tasks.find(t=>t.task_id===r.task_id)?.name||r.task_id);
      button.title=r.run_id;
      button.append(el('small',null,r.run_id));
      button.append(el('small',null,new Date(r.started_at*1000).toLocaleString()+' · '+({passed:'已达标',failed:'未达标',error:'出错'}[r.status]||r.status)));
      button.onclick=()=>watchHistory(r.run_id);
      const del=el('button','del','删除');
      del.type='button';del.title='删除这条实验记录（含轨迹与报告文件）';
      del.setAttribute('aria-label','删除实验记录 '+r.run_id);
      del.onclick=()=>removeRun(r.run_id,del);
      item.append(pick,button,del);
      box.append(item);
    }
    updateBatch();
  }catch(e){message('历史记录读取失败，可点击刷新重试。',true)}
}
async function loadStats(){
  try{
    const s=await api('/api/stats');
    $('stats').textContent='配方库 '+s.recipes+' · 归因库 '+s.attributions;
  }catch(_){$('stats').textContent='配方库 — · 归因库 —'}
}

$('start').onclick=async()=>{
  if(busy)return;
  busy=true;$('start').disabled=true;message('正在提交实验…');
  try{
    const r=await api('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      task_id:$('task').value,
      use_llm:$('llm').checked,
      recipe_lookup:$('recipe').checked,
      transfer_lookup:true,
      goal:$('goal').value,
    })});
    message('实验已开始，过程通过 SSE 实时推送。');
    watchLive(r.run_id,$('task').value);
  }catch(e){
    message('未能启动实验：'+e.message,true);
    busy=false;$('start').disabled=false;
  }
};
$('task').onchange=()=>{
  const t=tasks.find(x=>x.task_id===$('task').value);
  $('tasknote').textContent=t?.notes||'';
  target={threshold:t?Number(t.threshold):null,higher:t?!!t.higher_is_better:true};
  metricName=t?.metric||'';
  $('target').textContent=t?(target.higher?'≥ ':'≤ ')+target.threshold:'—';
  updateMetricLabels();
  schedulePreview();
};
$('goal').addEventListener('input',schedulePreview);
$('refresh').onclick=()=>{history();loadStats()};
$('batchdel').onclick=removeSelected;
$('pickall').onchange=()=>{
  const on=$('pickall').checked;
  for(const cb of $('history').querySelectorAll('input.pick')){
    cb.checked=on;
    on?selected.add(cb.dataset.rid):selected.delete(cb.dataset.rid);
  }
  updateBatch();
};

const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
if(SpeechRecognition){
  const recognition=new SpeechRecognition();
  recognition.lang='zh-CN';recognition.interimResults=false;recognition.maxAlternatives=1;
  recognition.onresult=event=>{
    const text=[...event.results].map(item=>item[0].transcript).join('').trim();
    if(text){$('goal').value=($('goal').value+' '+text).trim();schedulePreview()}
  };
  recognition.onerror=()=>message('语音识别不可用，请直接输入目标。',true);
  recognition.onend=()=>$('mic').classList.remove('on');
  $('mic').hidden=false;
  $('mic').onclick=()=>{
    if($('mic').classList.contains('on')){recognition.stop();return}
    $('mic').classList.add('on');
    try{recognition.start()}catch(_){$('mic').classList.remove('on')}
  };
}

(async()=>{
  try{
    try{
      const health=await api('/api/health');
      const box=$('llm');
      box.checked=!!health.llm_enabled;
      box.disabled=!health.llm_enabled;
      $('llmhint').textContent=health.llm_enabled
        ?'已检测到大模型配置；调用失败或输出不合规时会自动回退。'
        :'未检测到 DEEPSEEK_API_KEY，将使用模板规划与规则归因。';
    }catch(_){}
    tasks=await api('/api/tasks');
    const select=$('task');
    select.replaceChildren();
    for(const t of tasks){
      const option=document.createElement('option');
      option.value=t.task_id;option.textContent=t.name;select.append(option);
    }
    if(tasks.some(t=>t.task_id==='t1_demo'))select.value='t1_demo';
    select.disabled=false;$('start').disabled=!tasks.length;
    const current=tasks.find(t=>t.task_id===select.value);
    $('tasknote').textContent=current?.notes||'';
    target={threshold:current?Number(current.threshold):null,higher:current?!!current.higher_is_better:true};
    metricName=current?.metric||'';
    $('target').textContent=current?(target.higher?'≥ ':'≤ ')+target.threshold:'—';
    updateMetricLabels();
    schedulePreview();
    await history();
    loadStats();
  }catch(e){message('无法连接后端，请确认服务已启动后刷新页面。',true)}
})();
