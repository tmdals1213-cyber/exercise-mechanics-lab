"""Keep monitor wording consistent with four-step qualification and storage."""
from jump_monitor_data import ROOT

def main():
    path=ROOT/'scripts/jump_landing_live.html';text=path.read_text(encoding='utf-8')
    text=text.replace('넘어지는 시도도 끝까지 저장합니다. 안정 후보는 0.5·0.25·0.125ms에서 다시 계산해 구분합니다.','실패 결과와 파라미터를 보존합니다. 최종 후보는 0.5·0.25·0.125·0.0625ms 모두에서 3.000m 이상·안정 착지를 통과해야 합니다.')
    text=text.replace('안정 후보는 0.5/0.25/0.125ms에서 모두 다시 확인합니다.','최종 후보는 0.5/0.25/0.125/0.0625ms에서 모두 다시 확인합니다.')
    text=text.replace('모든 실패 시도의 원본 동작·접촉력·점수·파라미터를 로컬에 보존합니다.','모든 시행의 점수·파라미터·가벼운 이력을 보존합니다. 실패 원본 모델·동작 배열은 작업자별 최근24개와 영상/검증에 필요한 사례를 보존하며 나머지는 정리합니다.')
    text=text.replace("'0.5 / 0.25 / 0.125ms 검증'","'0.5 / 0.25 / 0.125 / 0.0625ms 검증'")
    path.write_text(text,encoding='utf-8');(ROOT/'deliverables/jump_landing_live/index.html').write_text(text,encoding='utf-8');print('MONITOR_STRICT_FOUR_STEP_SYNCED',flush=True)

if __name__=='__main__':main()
