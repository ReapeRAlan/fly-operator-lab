"""Render showcase videos: the game, the spiking connectome and the decision, tick by tick.

Every frame is one 50 ms decision of the simulation, so the video plays at real time when
encoded at 20 fps. Nothing is re-simulated here: it only draws what record_showcase.py stored.
"""
from pathlib import Path
import sys,json,math,argparse,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pyarrow.feather as feather
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import imageio_ffmpeg

INK=(9,20,28);PANEL=(14,28,38);ACCENT=(16,185,166);WARN=(245,158,11);BAD=(239,68,68);TEXT=(233,242,245);MUTED=(125,145,158)
FAMILY_COLOR={'descending_neuron':(16,230,200),'visual_projection':(96,165,250),'ol_intrinsic':(56,110,170),
              'cb_intrinsic':(150,120,220),'vnc_intrinsic':(120,180,140),'vnc_motor':(250,200,90),
              'ascending_neuron':(240,140,180),'cb_motor':(250,200,90)}
DEFAULT_COLOR=(120,140,160)

def font(size,bold=False):
    for name in (('seguibl.ttf','segoeuib.ttf') if bold else ('segoeui.ttf','arial.ttf')):
        try:return ImageFont.truetype(name,size)
        except OSError:continue
    return ImageFont.load_default(size)

def neuron_geometry():
    df=feather.read_table(ROOT/'data/processed/neurons.feather',columns=['somaLocation','superclass']).to_pandas()
    mask=df['somaLocation'].notna().to_numpy()
    xyz=np.stack(df.loc[mask,'somaLocation'].to_numpy()).astype(np.float32)
    families=df.loc[mask,'superclass'].fillna('other').to_numpy()
    colors=np.array([FAMILY_COLOR.get(f,DEFAULT_COLOR) for f in families],np.float32)
    centre=np.median(xyz,0)
    points=(xyz-centre)
    points=points[:,[0,2,1]];points[:,1]*=-1 # x right, y up (brain over nerve cord), z depth
    # Fill the frame vertically: the central nervous system is long and narrow.
    points/=np.percentile(np.abs(points[:,1]),99.2)
    index=np.full(len(df),-1,np.int64);index[np.flatnonzero(mask)]=np.arange(mask.sum())
    return points,colors,index,families

def project(points,angle,width,height,zoom=.92):
    c,s=math.cos(angle),math.sin(angle)
    x=points[:,0]*c+points[:,2]*s;z=-points[:,0]*s+points[:,2]*c;y=points[:,1]
    depth=(z+1.4)/2.8
    perspective=1/(1.9-z*.55)
    px=(x*perspective*zoom*.5+.5)*width;py=(y*perspective*zoom*.5+.5)*height
    return px,py,np.clip(depth,0,1)

def splat(canvas,px,py,colors,weights,spread=1):
    """Additive point splat; spread>1 draws a small square so sparse samples still read as tissue."""
    height,width=canvas.shape[:2];view=canvas.reshape(-1,3)
    offsets=[(0,0)] if spread<=1 else [(dx,dy) for dx in range(spread) for dy in range(spread)]
    share=1./len(offsets)
    for dx,dy in offsets:
        xi=np.rint(px+dx).astype(np.int32);yi=np.rint(py+dy).astype(np.int32)
        keep=(xi>=0)&(xi<width)&(yi>=0)&(yi<height)
        np.add.at(view,yi[keep]*width+xi[keep],colors[keep]*(weights[keep]*share)[:,None])

def brain_frame(size,points,colors,active_index,active_strength,angle,background_sample,descending=None,zoom=1.95):
    width,height=size
    canvas=np.zeros((height,width,3),np.float32)
    px,py,depth=project(points,angle,width,height,zoom)
    # anatomy: a dim nebula so the shape of the brain is always visible
    splat(canvas,px[background_sample],py[background_sample],colors[background_sample]/255.,
          (.16+.34*depth)[background_sample],spread=2)
    glow=np.zeros_like(canvas)
    if len(active_index):
        strength=np.clip(active_strength,0,4)/4.
        splat(glow,px[active_index],py[active_index],colors[active_index]/255.,
              .28+1.1*strength*(.45+.55*depth[active_index]),spread=2)
        if descending is not None:
            motor=active_index[descending[active_index]]
            if len(motor):
                splat(glow,px[motor],py[motor],np.tile(np.array([[.15,1.,.85]],np.float32),(len(motor),1)),
                      np.full(len(motor),2.6,np.float32),spread=3)
    base=Image.fromarray(np.clip(canvas*255,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1))
    halo=Image.fromarray(np.clip(glow*255,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5.5))
    core=Image.fromarray(np.clip(glow*255*1.4,0,255).astype(np.uint8))
    stack=np.asarray(base,np.float32)*1.35+np.asarray(halo,np.float32)*1.5+np.asarray(core,np.float32)*.35
    return Image.fromarray(np.clip(stack,0,255).astype(np.uint8))

def action_window(files,samples=16,half_width=160,aspect=16/9):
    """Follow the operator: the only thing that moves between consecutive frames inside the map.

    The head-up display (mission clock, portraits) also changes, so the search ignores the
    outer band of the capture and uses the median of the per-frame motion centroids.
    """
    if len(files)<4:return None
    picks=[files[int(i)] for i in np.linspace(0,len(files)-1,min(samples,len(files)))]
    frames=[np.asarray(Image.open(path).convert('L'),np.float32) for path in picks]
    height,width=frames[0].shape
    x0,x1=int(width*.10),int(width*.95);y0,y1=int(height*.05),int(height*.86)
    centroids=[]
    for previous,current in zip(frames,frames[1:]):
        motion=np.abs(current-previous)[y0:y1,x0:x1]
        if motion.max()<6:continue
        ys,xs=np.nonzero(motion>=max(np.percentile(motion,99.9),6))
        if len(xs)<8:continue
        centroids.append((np.median(xs)+x0,np.median(ys)+y0))
    if not centroids:return None
    cx=float(np.median([c[0] for c in centroids]));cy=float(np.median([c[1] for c in centroids]))
    spread=max(np.percentile([c[0] for c in centroids],90)-np.percentile([c[0] for c in centroids],10),
               np.percentile([c[1] for c in centroids],90)-np.percentile([c[1] for c in centroids],10))
    half_w=float(np.clip(spread*.75+half_width,230,width/2));half_h=half_w/aspect
    left=int(np.clip(cx-half_w,0,width-2*half_w));top=int(np.clip(cy-half_h,0,height-2*half_h))
    return (left,top,int(left+2*half_w),int(top+2*half_h))

def text(draw,xy,value,size=28,color=TEXT,bold=False,anchor=None):
    draw.text(xy,value,font=font(size,bold),fill=color,anchor=anchor)

def bar(draw,box,fraction,color=ACCENT,background=(30,48,60)):
    x0,y0,x1,y1=box
    draw.rounded_rectangle(box,radius=(y1-y0)//2,fill=background)
    if fraction>0:draw.rounded_rectangle((x0,y0,x0+max(6,(x1-x0)*min(1,fraction)),y1),radius=(y1-y0)//2,fill=color)

def load_run(folder):
    folder=Path(folder)
    trace=[json.loads(l) for l in (folder/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=json.loads((folder/'summary.json').read_text())
    archive=np.load(folder/'spikes.npz')
    return {'trace':trace,'summary':summary,'indices':archive['indices'],'counts':archive['counts'],'offsets':archive['offsets'],'folder':folder}

def game_frames(summary):
    directory=summary.get('capture_directory')
    if not directory:return []
    files=sorted(Path(directory).glob('*.png'))
    return files

def encoder(path,width,height,fps):
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    command=[ffmpeg,'-y','-f','rawvideo','-vcodec','rawvideo','-s',f'{width}x{height}','-pix_fmt','rgb24','-r',str(fps),
             '-i','-','-an','-vcodec','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    return subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def card(size,lines,seconds,fps,accent=ACCENT):
    width,height=size
    image=Image.new('RGB',size,INK);draw=ImageDraw.Draw(image)
    y=height//2-len(lines)*60
    for line,style in lines:
        sizes={'title':72,'big':54,'body':34,'small':26}[style]
        text(draw,(width//2,y),line,size=sizes,color=accent if style=='title' else TEXT,bold=style in('title','big'),anchor='mm')
        y+=int(sizes*1.7)
    for _ in range(int(seconds*fps)):yield image

def active_of(run,tick,index):
    """Neurons that spiked in that 50 ms tick, with their spike counts."""
    start,end=run['offsets'][tick],run['offsets'][tick+1]
    neurons=index[run['indices'][start:end]];strength=run['counts'][start:end].astype(np.float32)
    keep=neurons>=0
    return neurons[keep],strength[keep]

def compose_vertical(run,points,colors,index,background_sample,descending,size=(1080,1920),start_angle=0.):
    width,height=size;trace=run['trace'];summary=run['summary']
    shots=game_frames(summary)
    window=action_window(shots)
    game_height=int(width*(window[3]-window[1])/(window[2]-window[0])) if window else 620
    game_height=min(game_height,700)
    brain_size=(width,height-150-game_height-372)
    for tick,row in enumerate(trace):
        image=Image.new('RGB',size,INK);draw=ImageDraw.Draw(image)
        # header
        draw.rectangle((0,0,width,150),fill=PANEL)
        text(draw,(46,38),'FLY OPERATOR · MaleCNS v1.0',size=26,color=MUTED)
        text(draw,(46,76),summary['label'].replace('DESPUES','DESPUÉS'),size=46,color=ACCENT if 'DESPUES' in summary['label'] else WARN,bold=True)
        text(draw,(width-46,60),f"{row['sim_time_ms']/1000:5.2f} s",size=40,color=TEXT,bold=True,anchor='ra')
        text(draw,(width-46,108),'tiempo del juego',size=22,color=MUTED,anchor='ra')
        # game
        top=150
        frame_index=min(len(shots)-1,row.get('capture_frame') if row.get('capture_frame') is not None else tick)
        if shots and frame_index>=0:
            shot=Image.open(shots[frame_index]).convert('RGB')
            if window:shot=shot.crop(window)
            shot=shot.resize((width,game_height),Image.LANCZOS)
            image.paste(shot,(0,top))
        top+=game_height
        # brain
        angle=start_angle+tick*0.035
        neurons,strength=active_of(run,tick,index)
        brain=brain_frame(brain_size,points,colors,neurons,strength,angle,background_sample,descending)
        image.paste(brain,(0,top))
        draw=ImageDraw.Draw(image)
        text(draw,(46,top+14),'CEREBRO COMPLETO · 166,700 neuronas',size=26,color=MUTED)
        text(draw,(width-46,top+14),f"{row['spikes']:,} impulsos en 50 ms".replace(',','.'),size=30,color=ACCENT,bold=True,anchor='ra')
        text(draw,(46,top+brain_size[1]-44),'verde brillante: neuronas descendentes (la orden motora)',size=22,color=MUTED)
        # decision panel
        panel=top+brain_size[1]
        draw.rectangle((0,panel,width,height),fill=PANEL)
        text(draw,(46,panel+24),'LO QUE DECIDEN LAS NEURONAS DESCENDENTES',size=24,color=MUTED)
        text(draw,(46,panel+58),row['action'].upper(),size=56,color=TEXT,bold=True)
        y=panel+152
        for option in row['top_actions'][:3]:
            text(draw,(46,y),option['label'][:22],size=26,color=MUTED)
            bar(draw,(430,y+4,width-150,y+30),option['p'])
            text(draw,(width-46,y),f"{option['p']*100:3.0f}%",size=26,color=TEXT,anchor='ra')
            y+=48
        text(draw,(46,height-62),f"distancia al objetivo {row['goal_distance']:.2f} m",size=30,color=MUTED)
        if row['success']:text(draw,(width-46,height-66),'OBJETIVO CUMPLIDO',size=34,color=ACCENT,bold=True,anchor='ra')
        elif row['truncated']:text(draw,(width-46,height-66),'SE ACABÓ EL TIEMPO',size=34,color=BAD,bold=True,anchor='ra')
        yield image

def compose_brain_only(run,points,colors,index,background_sample,descending,size=(1080,1920),start_angle=.6):
    """Full-frame connectome: what the 50 ms of thinking behind each decision looks like."""
    width,height=size;trace=run['trace'];summary=run['summary']
    for tick,row in enumerate(trace):
        angle=start_angle+tick*0.045
        neurons,strength=active_of(run,tick,index)
        image=brain_frame((width,height),points,colors,neurons,strength,angle,background_sample,descending,zoom=2.15)
        draw=ImageDraw.Draw(image)
        text(draw,(60,70),'CEREBRO COMPLETO DE DROSOPHILA',size=34,color=TEXT,bold=True)
        text(draw,(60,118),'MaleCNS v1.0 · 166,700 neuronas · 25,582,938 conexiones',size=26,color=MUTED)
        text(draw,(60,height-210),f"{row['spikes']:,} impulsos".replace(',','.'),size=54,color=ACCENT,bold=True)
        text(draw,(60,height-146),f"en 50 ms · {row['active_neurons']:,} neuronas activas".replace(',','.'),size=28,color=MUTED)
        text(draw,(width-60,height-210),row['action'].upper(),size=40,color=TEXT,bold=True,anchor='ra')
        text(draw,(width-60,height-152),'lo que ordena al operador',size=24,color=MUTED,anchor='ra')
        bar(draw,(60,height-90,width-60,height-66),(tick+1)/len(trace),color=ACCENT)
        yield image

def compose_side_by_side(left,right,points,colors,index,background_sample,descending,size=(1920,1080)):
    """Both runs of the same mission playing at the same time."""
    width,height=size;half=width//2
    length=max(len(left['trace']),len(right['trace']))
    shots={'left':game_frames(left['summary']),'right':game_frames(right['summary'])}
    windows={side:action_window(files) for side,files in shots.items()}
    game_height=int(half*9/16)
    for tick in range(length):
        image=Image.new('RGB',size,INK);draw=ImageDraw.Draw(image)
        for side,run,x0 in (('left',left,0),('right',right,half)):
            row=run['trace'][min(tick,len(run['trace'])-1)];done=tick>=len(run['trace'])
            colour=WARN if 'SIN' in run['summary']['label'] else ACCENT
            draw.rectangle((x0,0,x0+half,96),fill=PANEL)
            text(draw,(x0+36,26),run['summary']['label'].replace('DESPUES','DESPUÉS'),size=40,color=colour,bold=True)
            text(draw,(x0+half-36,30),f"{row['sim_time_ms']/1000:5.2f} s",size=34,color=MUTED,anchor='ra')
            files=shots[side]
            if files:
                frame_index=min(len(files)-1,row.get('capture_frame') if row.get('capture_frame') is not None else tick)
                shot=Image.open(files[max(0,frame_index)]).convert('RGB')
                if windows[side]:shot=shot.crop(windows[side])
                image.paste(shot.resize((half,game_height),Image.LANCZOS),(x0,96))
            neurons,strength=active_of(run,min(tick,len(run['trace'])-1),index)
            brain=brain_frame((half,height-96-game_height-120),points,colors,neurons,strength,.3+tick*.03,background_sample,descending)
            image.paste(brain,(x0,96+game_height))
            draw=ImageDraw.Draw(image)
            text(draw,(x0+36,height-96),row['action'].upper(),size=34,color=TEXT,bold=True)
            text(draw,(x0+36,height-52),f"a {row['goal_distance']:.2f} m del objetivo",size=26,color=MUTED)
            if done and run['summary']['success']:text(draw,(x0+half-36,height-96),'OBJETIVO CUMPLIDO',size=34,color=ACCENT,bold=True,anchor='ra')
            elif done:text(draw,(x0+half-36,height-96),'SIN LOGRARLO',size=34,color=BAD,bold=True,anchor='ra')
        draw.line((half,0,half,height),fill=(30,48,60),width=3)
        yield image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='work/showcase')
    parser.add_argument('--output',default='D:/Desktop/Videos Fly Operator')
    parser.add_argument('--fps',type=int,default=20)
    parser.add_argument('--background',type=int,default=26000)
    args=parser.parse_args()
    source=ROOT/args.input;out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    points,colors,index,families=neuron_geometry()
    descending=(families=='descending_neuron')
    rng=np.random.default_rng(7)
    background_sample=rng.choice(len(points),size=min(args.background,len(points)),replace=False)
    runs=[load_run(f) for f in sorted(source.glob('*__*')) if (f/'summary.json').exists()]
    runs.sort(key=lambda r:(r['summary']['mission'],'SIN' not in r['summary']['label']))
    if not runs:raise SystemExit('No recorded runs in '+str(source))
    print('runs:',[r['summary']['label']+' '+r['summary']['mission'] for r in runs])
    size=(1080,1920)
    writer=encoder(out/'fly_aprende_vertical.mp4',*size,args.fps)
    def push(frames):
        for frame in frames:writer.stdin.write(np.asarray(frame,np.uint8).tobytes())
    push(card(size,[('EL CEREBRO DE UNA MOSCA','title'),('aprende a jugar','big'),('Door Kickers 2','big'),
                    ('conectoma completo MaleCNS v1.0','body'),('166,700 neuronas · 25,582,938 conexiones','small')],2.6,args.fps))
    for run in runs:
        summary=run['summary']
        push(card(size,[(summary['label'].replace('DESPUES','DESPUÉS'),'title'),(f"misión {summary['mission'].replace('fly_','')}",'body'),
                        ('tiempo real · 1 cuadro = 1 decisión de 50 ms','small')],1.6,args.fps,
                  accent=ACCENT if 'DESPUES' in summary['label'] else WARN))
        push(compose_vertical(run,points,colors,index,background_sample,descending,size))
        result='objetivo cumplido' if summary['success'] else 'no lo logró'
        push(card(size,[(result.upper(),'title'),(f"{summary['ticks']} decisiones · {summary['ticks']*50/1000:.1f} s de juego",'body'),
                        (f"distancia final {summary['final_goal_distance']:.2f} m",'small')],1.4,args.fps,
                  accent=ACCENT if summary['success'] else BAD))
    push(card(size,[('MISMO CEREBRO','title'),('misma máscara, misma misión','body'),
                    ('lo único que cambió','body'),('son los pesos aprendidos','big')],2.4,args.fps))
    writer.stdin.close();writer.wait()
    print('wrote',out/'fly_aprende_vertical.mp4',flush=True)

    # 2) the connectome alone, as a loop
    writer=encoder(out/'cerebro_3d_vertical.mp4',*size,args.fps)
    angle=.6
    for run in runs:
        for frame in compose_brain_only(run,points,colors,index,background_sample,descending,size,start_angle=angle):
            writer.stdin.write(np.asarray(frame,np.uint8).tobytes())
        angle+=len(run['trace'])*0.045
    writer.stdin.close();writer.wait()
    print('wrote',out/'cerebro_3d_vertical.mp4',flush=True)

    # 3) the same mission side by side
    pairs={}
    for run in runs:pairs.setdefault(run['summary']['mission'],{})['SIN' if 'SIN' in run['summary']['label'] else 'OK']=run
    duo=next((p for p in pairs.values() if len(p)==2),None)
    if duo:
        wide=(1920,1080);writer=encoder(out/'antes_despues_horizontal.mp4',*wide,args.fps)
        for frame in compose_side_by_side(duo['SIN'],duo['OK'],points,colors,index,background_sample,descending,wide):
            writer.stdin.write(np.asarray(frame,np.uint8).tobytes())
        writer.stdin.close();writer.wait()
        print('wrote',out/'antes_despues_horizontal.mp4',flush=True)

if __name__=='__main__':main()
