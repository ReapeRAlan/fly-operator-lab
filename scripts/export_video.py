"""Export captured game frames at 30 fps, with episode/time/neural telemetry."""
from pathlib import Path
import json,sys,subprocess,collections,hashlib,argparse,itertools
ROOT=Path(__file__).resolve().parents[1]

def asstime(seconds):
    ticks=int(seconds*100+1e-6);return f'{ticks//360000}:{ticks//6000%60:02}:{ticks//100%60:02}.{ticks%100:02}'

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--frames',type=int)
    args=p.parse_args();directories=[Path(x).resolve() for x in args.runs];directory=directories[0]
    rows=[];offset=0;wall_total=0
    for folder in directories:
        summary=json.loads((folder/'summary.json').read_text())
        if not summary['passed']:raise RuntimeError('Cannot export an incomplete run')
        part=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
        for row in part:row['episode_index']+=offset
        rows.extend(part);offset+=len(summary['episodes']);wall_total+=summary['wall_seconds']
    if args.frames:
        if len(rows)<args.frames:raise RuntimeError('Not enough captured frames; record a supplemental episode')
        rows=rows[:args.frames]
    episode_count=max(r['episode_index'] for r in rows)+1
    target=ROOT/'outputs/Fly_Operator_MaleCNS_1080p30.mp4'
    subtitle=directory/'overlay.ass'
    header='''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Arial,26,&H00FFFFFF,&H00FFFFFF,&H00101820,&H00101820,-1,0,0,0,100,100,0,0,3,8,0,7,24,24,18,1
Style: Telemetry,Arial,23,&H00FFFFFF,&H00FFFFFF,&H00101820,&H00101820,0,0,0,0,100,100,0,0,3,8,0,1,24,24,24,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    lines=[header];dur=len(rows)/30
    lines.append(f'Dialogue: 0,{asstime(0)},{asstime(dur)},Title,,0,0,0,,FLY OPERATOR | MaleCNS v1.0 | 166,700 neuronas | 25,582,938 conexiones\n')
    names={'move':'mover','stop':'detener','turn_left':'girar izquierda','turn_right':'girar derecha','aim':'apuntar','fire':'disparar','reload':'recargar'}
    for index,row in enumerate(rows):
        obs=row['result'];mode={'normal':'Entradas normales','altered_inputs':'Entradas alteradas','disconnected_outputs':'Salidas neuronales desconectadas'}[row['mode']]
        text=f'Episodio {row["episode_index"]+1}/{episode_count} | {mode} | Tiempo simulado {obs["sim_time_ms"]/1000:.2f} s'
        text+=f'\\NAccion: {names[row["executed_command"]["action"]]} | {row["activity"]["spikes"]:,} disparos neuronales / {row["activity"]["simulated_ms"]} ms | Esperas de calculo eliminadas'
        lines.append(f'Dialogue: 1,{asstime(index/30)},{asstime((index+1)/30)},Telemetry,,0,0,0,,{text}\n')
    subtitle.write_text(''.join(lines),encoding='utf-8')
    ffmpeg=ROOT/'.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    if not ffmpeg.exists():
        import imageio_ffmpeg
        ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe())
    cmd=[str(ffmpeg),'-hide_banner','-loglevel','warning','-y','-f','image2pipe','-framerate','30','-vcodec','png','-i','-',
         '-vf','scale=1920:1080:flags=lanczos,subtitles=overlay.ass','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    with (directory/'ffmpeg.log').open('w') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log,cwd=directory)
        try:
            for row in rows:
                obs=row['result'];idx=obs['capture_frame'];folder=Path(obs['capture_directory'])
                frame=obs['frame']
                if not frame['saved'] or frame['client_sim_time_ms']!=frame['sim_time_ms']:raise RuntimeError('Unsynchronized frame')
                process.stdin.write((folder/f'frame_{idx:06d}.png').read_bytes())
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg failed; inspect log')
    expected=sum(r['activity']['simulated_ms'] for r in rows)/1000
    if abs(dur-expected)>1/30:raise RuntimeError('Video duration differs from total simulated duration')
    data={'file':str(target),'frames':len(rows),'fps':30,'width':1920,'height':1080,'duration_seconds':dur,'total_simulated_seconds':expected,'episodes':episode_count,'source_runs':[str(x) for x in directories],'source_sizes':sorted({(r['result']['frame']['width'],r['result']['frame']['height']) for r in rows}),'upscaled':True,'wall_run_seconds':wall_total,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'telemetry':'Overlay is experimental telemetry; gameplay frames are direct OpenGL captures. Episode cuts exclude reload and computation waits. The final episode may be cut at the requested frame count.'}
    (ROOT/'outputs/video_validation.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
if __name__=='__main__':main()
