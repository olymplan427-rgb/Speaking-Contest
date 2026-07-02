"""
수학말하기대회 AI 심사 — 로컬 Vision 서버
실행: python server.py
포트: 8080
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess, tempfile, os, base64, json, re

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/extract-frames', methods=['POST'])
def extract_frames():
    data = request.json or {}
    url = (data.get('url') or '').strip()
    duration_hint = float(data.get('duration_sec') or 0)

    if not url:
        return jsonify({'error': 'URL이 없습니다.'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, 'video.mp4')

        # ━━ 1. 영상 다운로드 (yt-dlp → ffmpeg 직접 순서) ━━
        downloaded = False
        try:
            r = subprocess.run([
                'yt-dlp',
                '-f', 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best[height<=720]/best',
                '--merge-output-format', 'mp4',
                '-o', video_path,
                '--no-playlist',
                '--socket-timeout', '30',
                url
            ], capture_output=True, text=True, timeout=180)
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                downloaded = True
        except subprocess.TimeoutExpired:
            pass

        if not downloaded:
            # 직접 MP4 URL이면 ffmpeg로 바로 받기
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', url,
                    '-t', '600', '-c', 'copy', video_path
                ], capture_output=True, timeout=60)
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    downloaded = True
            except subprocess.TimeoutExpired:
                pass

        if not downloaded:
            return jsonify({'error': '영상 다운로드 실패. URL을 확인하거나 직접 MP4 파일을 제공해주세요.'}), 500

        # ━━ 2. 실제 재생시간 감지 ━━
        actual_duration = duration_hint
        try:
            probe = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', video_path
            ], capture_output=True, text=True, timeout=15)
            info = json.loads(probe.stdout)
            # format duration가 가장 신뢰성 높음
            fmt_dur = float(info.get('format', {}).get('duration', 0))
            if fmt_dur > 0:
                actual_duration = fmt_dur
            else:
                for s in info.get('streams', []):
                    if s.get('codec_type') == 'video':
                        d = float(s.get('duration', 0))
                        if d > 0:
                            actual_duration = d
                        break
        except Exception:
            pass

        if actual_duration <= 0:
            actual_duration = 180

        # ━━ 3. 공백구간 감지 (끝부분 4개 지점 밝기 체크) ━━
        effective_duration = actual_duration
        checkpoints = [0.95, 0.85, 0.75, 0.65]
        for ratio in checkpoints:
            t = actual_duration * ratio
            snap = os.path.join(tmpdir, 'snap.jpg')
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(t), '-i', video_path,
                '-vframes', '1', '-vf', 'scale=64:36', '-q:v', '5', snap
            ], capture_output=True, timeout=8)
            if not os.path.exists(snap):
                break
            # 밝기 계산 (ffprobe signalstats)
            r2 = subprocess.run([
                'ffprobe', '-v', 'quiet', '-f', 'lavfi',
                f'-i', f'movie={snap},signalstats',
                '-show_entries', 'frame_tags=lavfi.signalstats.YAVG',
                '-print_format', 'json'
            ], capture_output=True, text=True, timeout=8)
            brightness = 0
            try:
                tags = json.loads(r2.stdout).get('frames', [{}])[0].get('tags', {})
                brightness = float(tags.get('lavfi.signalstats.YAVG', 0))
            except Exception:
                brightness = 10  # 감지 불가 시 유효로 간주
            if brightness > 8:
                effective_duration = min(actual_duration * ratio + actual_duration * 0.1, actual_duration)
                break
            effective_duration = actual_duration * ratio

        # ━━ 4. 3구간 프레임 추출 (도입 15% / 전개 50% / 마무리 82%) ━━
        timepoints = [effective_duration * 0.15, effective_duration * 0.50, effective_duration * 0.82]
        labels = ['도입부', '전개부', '마무리']
        frames = []

        for i, t in enumerate(timepoints):
            fp = os.path.join(tmpdir, f'frame_{i}.jpg')
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(t), '-i', video_path,
                '-vframes', '1', '-vf', 'scale=960:-1', '-q:v', '3', fp
            ], capture_output=True, timeout=15)
            if os.path.exists(fp) and os.path.getsize(fp) > 0:
                with open(fp, 'rb') as f:
                    frames.append(base64.b64encode(f.read()).decode())
            else:
                frames.append(None)

        success = sum(1 for f in frames if f)
        return jsonify({
            'frames': frames,
            'labels': labels,
            'actual_duration': round(actual_duration, 1),
            'effective_duration': round(effective_duration, 1),
            'success_count': success
        })


@app.route('/extract-transcript', methods=['POST'])
def extract_transcript():
    """YouTube 자동자막(한국어) 추출 → 타임스탬프 + 전체 텍스트 반환"""
    data = request.json or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'URL이 없습니다.'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        sub_base = os.path.join(tmpdir, 'sub')

        # ━━ 1. 한국어 자동자막 다운로드 (수동자막 우선, 없으면 자동생성) ━━
        result = subprocess.run([
            'yt-dlp',
            '--write-subs', '--write-auto-subs',
            '--sub-lang', 'ko',
            '--sub-format', 'vtt',
            '--skip-download',
            '--no-playlist',
            '-o', sub_base,
            url
        ], capture_output=True, text=True, timeout=60)

        # VTT 파일 찾기
        vtt_path = None
        for fn in os.listdir(tmpdir):
            if fn.endswith('.vtt'):
                vtt_path = os.path.join(tmpdir, fn)
                break

        if not vtt_path:
            return jsonify({'error': '자막을 찾을 수 없습니다. YouTube 자동자막이 없는 영상입니다.', 'segments': [], 'full_text': ''}), 200

        # ━━ 2. VTT 파싱 → 타임스탬프 세그먼트 ━━
        with open(vtt_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        segments = []
        seen_texts = set()
        # VTT 블록: 00:00:00.000 --> 00:00:00.000\n텍스트
        blocks = re.split(r'\n\n+', raw)
        for block in blocks:
            lines = block.strip().splitlines()
            # 타임스탬프 줄 찾기
            ts_line = next((l for l in lines if '-->' in l), None)
            if not ts_line:
                continue
            m = re.match(r'(\d+:\d+:\d+[\.,]\d+)\s*-->\s*(\d+:\d+:\d+[\.,]\d+)', ts_line)
            if not m:
                continue
            start_str = m.group(1).replace(',', '.')
            # 텍스트 줄: 타임스탬프 이후 줄, HTML 태그 제거
            text_lines = [l for l in lines if '-->' not in l and not l.strip().isdigit() and l.strip()]
            text = ' '.join(text_lines)
            text = re.sub(r'<[^>]+>', '', text).strip()  # HTML 태그 제거
            text = re.sub(r'\s+', ' ', text)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            # 타임스탬프 → 초
            parts = start_str.split(':')
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            segments.append({'time': round(secs, 1), 'time_str': f'{int(secs//60):02d}:{int(secs%60):02d}', 'text': text})

        full_text = ' '.join(s['text'] for s in segments)
        return jsonify({'segments': segments, 'full_text': full_text, 'count': len(segments)})


if __name__ == '__main__':
    print("=" * 50)
    print("  수학말하기대회 AI 심사 — Vision 서버")
    print("  http://localhost:8080")
    print("  종료: Ctrl+C")
    print("=" * 50)
    app.run(host='127.0.0.1', port=8080, debug=False)
