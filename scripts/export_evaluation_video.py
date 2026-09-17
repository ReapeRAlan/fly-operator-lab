"""Export captured evaluation frames at 1080p/30 using simulation timestamps."""
from pathlib import Path
import argparse,json,subprocess,sys
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',required=True);p.add_argument('--output',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    folder=Path(a.frames).resolve();allowed=(ROOT/'work/frames').resolve()
    if not folder.is_relative_to(allowed):raise ValueError('Expected laboratory capture directory')
    rows=[json.loads(l) for l in (folder/'timestamps.jsonl').read_text().splitlines()];assert rows and all(r['saved'] for r in rows)
    if any(b['sim_time_ms']<=a['sim_time_ms'] for a,b in zip(rows,rows[1:])):raise ValueError('Non-increasing simulation timestamps')
    concat=folder/'export.ffconcat';lines=['ffconcat version 1.0'];durations=[]
    for i,row in enumerate(rows):
        # The first capture covers time 0..first timestamp; later frames begin at their timestamp.
        duration=(rows[i+1]['sim_time_ms']-row['sim_time_ms'])/1000 if i<len(rows)-1 else (row['sim_time_ms']-rows[i-1]['sim_time_ms'])/1000 if i else .05
        durations.append(duration);lines +=[f"file 'frame_{row['frame']:06d}.png'",f'duration {duration:.6f}']
    lines +=[f"file 'frame_{rows[-1]['frame']:06d}.png'"];concat.write_text('\n'.join(lines))
    label=folder/'clip_label.txt';label.write_text(a.label,encoding='utf-8');output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    ff=imageio_ffmpeg.get_ffmpeg_exe();filter="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,drawbox=x=0:y=0:w=iw:h=85:color=black@0.7:t=fill,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':textfile='clip_label.txt':x=34:y=22:fontsize=24:fontcolor=white"
    command=[ff,'-y','-loglevel','error','-f','concat','-safe','0','-i','export.ffconcat','-vf',filter,'-t',str(sum(durations)),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(command,cwd=folder,check=True)
    verification=subprocess.run([ff,'-v','error','-i',str(output),'-f','null','-'],capture_output=True,text=True,check=True)
    manifest={'frames':len(rows),'source_width':rows[0]['width'],'source_height':rows[0]['height'],'source_decision_ms':50,'dimensions':[1920,1080],'fps':30,
      'simulated_duration_seconds':sum(durations),'label':a.label,'decoded_without_errors':True,'source_frames':str(folder),'output':str(output),
      'timing':'Frame holds at 30 fps; neural computation waiting time excluded. Native 20 fps decision captures, scaled to 1080p.'}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
