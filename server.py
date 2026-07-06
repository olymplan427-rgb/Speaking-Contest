"""
수학말하기대회 AI 심사 — 로컬 Vision 서버
실행: python server.py
포트: 8080

━━ NVIDIA NIM API 키 설정 ━━
아래 NVIDIA_API_KEY에 nvapi-... 키를 붙여넣으세요.
발급: https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr → Get API Key
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess, tempfile, os, base64, json, re

# ★ 여기에 NVIDIA NIM API 키를 입력하세요 ★
NVIDIA_API_KEY = "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"

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
    """음성 전사: 영상에서 오디오 추출 → NVIDIA NIM Parakeet ASR"""
    data = request.json or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'URL이 없습니다.'}), 400

    nvidia_key = NVIDIA_API_KEY.strip()
    if not nvidia_key or nvidia_key == 'nvapi-여기에붙여넣기':
        return jsonify({
            'error': 'NVIDIA API 키 미설정 — server.py 상단 NVIDIA_API_KEY에 nvapi-... 키를 입력 후 재시작하세요.',
            'segments': [], 'full_text': ''
        }), 200

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, 'video.mp4')
        audio_path = os.path.join(tmpdir, 'audio.mp3')

        # ━━ 1. 영상 다운로드 ━━
        print(f'[전사] 영상 다운로드 시작: {url}')
        downloaded = False
        try:
            r = subprocess.run([
                'yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best',
                '--merge-output-format', 'mp4',
                '--no-playlist', '--socket-timeout', '30',
                '-o', video_path, url
            ], capture_output=True, text=True, timeout=180)
            print(f'[전사] yt-dlp stdout: {r.stdout[-300:]}')
            print(f'[전사] yt-dlp stderr: {r.stderr[-300:]}')
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                downloaded = True
                print(f'[전사] yt-dlp 다운로드 성공: {os.path.getsize(video_path)} bytes')
        except Exception as e:
            print(f'[전사] yt-dlp 실패: {e}')

        if not downloaded:
            try:
                r2 = subprocess.run([
                    'ffmpeg', '-y', '-i', url, '-t', '600', '-c', 'copy', video_path
                ], capture_output=True, text=True, timeout=60)
                print(f'[전사] ffmpeg stderr: {r2.stderr[-300:]}')
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    downloaded = True
                    print(f'[전사] ffmpeg 다운로드 성공: {os.path.getsize(video_path)} bytes')
            except Exception as e:
                print(f'[전사] ffmpeg 실패: {e}')

        if not downloaded:
            print('[전사] 영상 다운로드 최종 실패')
            return jsonify({'error': '영상 다운로드 실패. URL을 확인하세요.', 'segments': [], 'full_text': ''}), 200

        # ━━ 2. 오디오 추출 ━━
        print('[전사] 오디오 추출 중...')
        r3 = subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-ar', '16000', '-ac', '1', '-b:a', '64k',
            audio_path
        ], capture_output=True, text=True, timeout=120)
        print(f'[전사] ffmpeg 오디오 stderr: {r3.stderr[-200:]}')

        if not (os.path.exists(audio_path) and os.path.getsize(audio_path) > 0):
            print('[전사] 오디오 추출 실패')
            return jsonify({'error': '오디오 추출 실패.', 'segments': [], 'full_text': ''}), 200
        print(f'[전사] 오디오 추출 성공: {os.path.getsize(audio_path)} bytes')

        # ━━ 3. NVIDIA NIM Parakeet ASR ━━
        print('[전사] NVIDIA Parakeet ASR 요청 중...')
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url='https://integrate.api.nvidia.com/v1',
                api_key=nvidia_key
            )
            with open(audio_path, 'rb') as f:
                response = client.audio.transcriptions.create(
                    model='nvidia/parakeet-1.1b-rnnt-multilingual-asr',
                    file=f,
                    response_format='verbose_json'
                )
            print(f'[전사] 응답 타입: {type(response)}')
            print(f'[전사] 응답 내용: {str(response)[:500]}')

            segs = []
            if hasattr(response, 'segments') and response.segments:
                for s in response.segments:
                    text = (s.text or '').strip()
                    if text:
                        segs.append({'time': round(float(s.start), 1), 'text': text})
                full_text = (response.text or '').strip() or ' '.join(s['text'] for s in segs)
            else:
                full_text = (response.text or '').strip()
                if full_text:
                    segs = [{'time': 0.0, 'text': full_text}]

            print(f'[전사] 완료: {len(segs)}개 세그먼트')
            return jsonify({'segments': segs, 'full_text': full_text, 'count': len(segs), 'method': 'nvidia-parakeet'})

        except Exception as e:
            import traceback
            print(f'[전사] NVIDIA 오류: {e}')
            print(traceback.format_exc())
            return jsonify({'error': f'NVIDIA Parakeet 오류: {str(e)}', 'segments': [], 'full_text': ''}), 200


if __name__ == '__main__':
    print("=" * 50)
    print("  수학말하기대회 AI 심사 — Vision 서버")
    print("  http://localhost:8080")
    print("  종료: Ctrl+C")
    print("=" * 50)
    app.run(host='127.0.0.1', port=8080, debug=False)
