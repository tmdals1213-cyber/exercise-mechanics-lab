"""Readable local reports generated from project documentation and computed data."""
from pathlib import Path
import html,json,re
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ohp_fullbody'
STYLE='body{background:#0d1219;color:#dce6ef;font:17px/1.8 "Malgun Gothic",sans-serif;margin:0;padding:32px}article{max-width:1000px;margin:auto}h1,h2{line-height:1.4;color:#6fddc4}h2{margin-top:38px}a{color:#80caff}p{margin:12px 0}pre{white-space:pre-wrap;background:#18212d;padding:20px;border-radius:12px}table{border-collapse:collapse;width:100%;margin:22px 0;font-size:15px}th,td{padding:10px;text-align:right;border-bottom:1px solid #334355}th:first-child,td:first-child{text-align:left}.note{color:#b0c1d0}'
def page(title,body):return f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style><article>{body}</article></html>'
def escape_links(text):
 escaped=html.escape(text)
 return re.sub(r'https://[^\s<>]+',lambda m:f'<a href="{m[0]}">{m[0]}</a>',escaped)
lines=[]
for line in (ROOT/'README.md').read_text('utf-8-sig').splitlines():
 if line.startswith('## '):lines.append('<h2>'+html.escape(line[3:])+'</h2>')
 elif line.startswith('# '):lines.append('<h1>'+html.escape(line[2:])+'</h1>')
 elif line.startswith('    '):lines.append('<pre>'+html.escape(line)+'</pre>')
 elif line:lines.append('<p>'+escape_links(line)+'</p>')
(OUT/'mechanics_basis.html').write_text(page('OHP 계산 근거와 한계',''.join(lines)),encoding='utf-8')
report=json.loads((OUT/'height_comparison.json').read_text('utf-8'))
body='<h1>165 · 180 · 195 cm — 같은 100 kg OHP</h1><p>같은 관절각·2초 상승, 길이에 비례한 그립·경로. 바벨의 순중력 일은 mgΔh입니다. 근육별 장력은 전신 역동역학과 힘 분담 최적화의 추정값입니다.</p>'
for policy in ['geometric','same_bmi','same_mass']:
 rows=[r for r in report['cases'] if r['policy']==policy]
 body+=f'<h2>{rows[0]["label"]}</h2><table><tr><th>키</th><th>체중</th><th>BMI</th><th>바 이동</th><th>바의 일</th><th>삼각근 최대합</th><th>삼두근 최대합</th><th>가정 근력 기준</th></tr>'
 for r in rows:
  body+=f'<tr><td>{r["height_cm"]:.0f} cm</td><td>{r["mass_kg"]:.1f} kg</td><td>{r["bmi"]:.1f}</td><td>{r["bar_travel_m"]*100:.1f} cm</td><td>{r["bar_gravitational_work_J"]:.1f} J</td><td>{r["peak_deltoid_sum_N"]/1000:.2f} kN</td><td>{r["peak_triceps_sum_N"]/1000:.2f} kN</td><td>{r["relative_reference_strength"]*100:.1f}%</td></tr>'
 body+='</table>'
body+='<h2>왜 짧은 팔이면 힘도 비례해서 줄어들지 않는가</h2><pre>외부 토크 ≈ 바벨 힘 × 외부 모멘트암\n근육 힘 ≈ 외부 토크 / 근육 모멘트암\n길이 배율 s를 두 모멘트암에 함께 적용하면: (s × 외부 토크) / (s × 근육 모멘트암)</pre><p>바벨의 정적 부하만 보면 공통 길이 배율이 소거됩니다. 전신에서는 신체 분절 무게와 가속도, 관절각 및 힘 분담이 더해져 달라집니다. 단순히 키 비율만큼 근육 장력이 감소한다는 결론은 성립하지 않습니다.</p>'
body+='<h2>근력 기준의 가정</h2><p>일정한 근육 질량 비율과 근섬유 길이 ∝ 키를 가정하여 기준 힘 비율을 질량 비율 / 키 비율로 두었습니다. BMI로 실제 근력을 예측한 값이 아닙니다. 동일 BMI·동일 체중 조건은 근육 내부 경로의 굵기를 재구축한 모델이 아니라 질량·기준 힘 민감도 분석입니다. 요구 장력에 근력 상한을 적용하지 않아, 이 동작을 실제 수행할 수 있다는 뜻도 아닙니다.</p>'
body+='<p><a href="mechanics_basis.html">계산 원리·검산·출처·한계 전체</a> · <a href="height_comparison.csv">계산 수치 CSV</a> · <a href="height_comparison.json">설정과 원시 요약 JSON</a></p>'
(OUT/'height_comparison_report.html').write_text(page('키와 OHP의 역학 비교',body),encoding='utf-8')
(OUT/'CREDITS.txt').write_text((ROOT/'assets/fullbody/CREDITS.md').read_text('utf-8'),encoding='utf-8')
if (OUT/'bar_joint_metrics.npz').exists():
 metrics=np.load(OUT/'bar_joint_metrics.npz');summary=json.loads((OUT/'bar_joint_metrics.json').read_text('utf-8'))
 body='<h1>바벨의 일·출력과 팔꿈치·손목 지표</h1><p>서로 다른 양을 구분합니다. 영상의 관절 값은 바벨 외력모멘트입니다. 모델 전달력과 순관절 토크는 아래 별도 표입니다.</p>'
 body+='<h2>일과 출력</h2><pre>손→바벨 누적 일 = 중력 위치에너지 증가 + 병진 운동에너지 증가\n순간 기계적 출력 = 손이 바벨에 가하는 힘 · 바벨 속도</pre>'
 body+=f'<p>상승 높이 {summary["bar_rise_m"]*100:.1f} cm, 최종 일 {summary["bar_final_hand_work_J"]:.1f} J, 2초 평균 출력 {summary["bar_average_ascent_power_W"]:.1f} W, 최대 출력 {summary["bar_peak_power_W"]:.1f} W. 정지 유지 중 출력은 0 W입니다. 힘이나 대사 에너지 소비가 0이라는 뜻은 아닙니다.</p>'
 body+='<h2>1. 바벨 외력모멘트 — 영상 표시</h2><p>관절 중심에서 손의 바벨 외력까지의 위치 벡터와 힘의 외적 크기입니다. 외력의 지렛대 효과이며, 근육 힘이나 관절면 압력이 아닙니다.</p>'
 body+='<h2>2. 순관절 토크 — 모델 좌표</h2><p>역동역학으로 해당 굽힘 좌표에 필요한 순토크를 계산합니다. 신체 분절의 중력·관성이 포함됩니다. 이 축의 토크가 작아도 다른 방향 힘과 관절 전달력은 남을 수 있습니다.</p>'
 body+='<h2>3. 모델 관절 전달력 — 근육 힘 포함</h2><p>손 또는 전완·손 전체의 절단면 뉴턴 평형으로 계산한 합력입니다. 할당된 근육 힘, 바벨 외력, 분절 중력과 COM 가속도를 포함합니다. 현재 근육 힘 분담, 고정 손가락, 커플링, 기준 근력 상한 미적용의 영향을 받으므로 실제 사람의 연골·인대 하중으로 해석하지 않습니다.</p>'
 body+='<table><tr><th>시간</th><th>바벨 일</th><th>출력</th><th>팔꿈치 외력모멘트</th><th>손목 외력모멘트</th><th>팔꿈치 모델 합력</th><th>손목 모델 합력</th></tr>'
 for index in [0,15,30,45,60,75]:
  ext=np.linalg.norm(metrics['bar_external_joint_moment_world_Nm'][index],axis=1);reaction=np.linalg.norm(metrics['joint_reaction_world_N'][index],axis=1)
  body+=f'<tr><td>{metrics["times"][index]:.2f} s</td><td>{metrics["bar_hand_work_J"][index]:.1f} J</td><td>{metrics["bar_power_W"][index]:.1f} W</td><td>{ext[0]:.1f} N·m</td><td>{ext[1]:.1f} N·m</td><td>{reaction[0]:.0f} N</td><td>{reaction[1]:.0f} N</td></tr>'
 body+='</table><p>관절 지표는 우측 기준입니다. 전달력 수치의 생리적 검증은 하지 않았습니다.</p>'
 body+=f'<h2>구현 검산</h2><p>출력 적분과 에너지 증가의 차이: {summary["work_power_integral_error_J"]:.3g} J. 건 절점 힘의 일반화 힘 투영 차이: {summary["nodal_virtual_work_max_error"]:.3g}. 원본 좌표에서 독립 MuJoCo 후처리 전달력과의 차이: {summary["native_joint_reaction_max_error_N"]:.3g} N. 이는 구현 검산이며 인체 실측 검증이 아닙니다.</p>'
 body+='<p><a href="bar_joint_metrics.csv">시계열 CSV</a> · <a href="bar_joint_metrics.json">정의·요약·한계 JSON</a> · <a href="mechanics_basis.html">전체 계산 방법</a></p>'
 (OUT/'bar_joint_report.html').write_text(page('일·출력·관절 부하의 구분',body),encoding='utf-8')
print('READABLE_REPORTS_PREPARED')
