#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>
#include <bcrypt.h>
#include <gl/GL.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <mutex>
#include <atomic>
#include <vector>
#include <deque>
#include <cmath>
#include "MinHook.h"
#include "json.hpp"
#include "build_profile.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using json=nlohmann::json;
namespace fs=std::filesystem;
static uintptr_t base;
static fs::path root;
static std::ofstream logfile;
static std::mutex mu;
static HANDLE replyEvent;
static json request,reply;
static bool pending=false, responding=false;
static std::atomic<bool> connected=false, active=false, stepping=false;
static std::atomic<bool> restartRequested=false;
static std::atomic<bool> loadRequested=false,deployRequested=false;
static uint64_t episode=0,sequence=0,virtualUs=0;
static uint32_t operatorId=0;
static int shotRequests=0,shotsAccepted=0,blockedAuto=0,reloadRequests=0;
static uintptr_t selected=0;
static bool clockSet=false, record=false;
static std::atomic<int> frameNo=0;
static std::atomic<int> captureRequested=0;
static int captureTime=0;
static unsigned presentIntervalMs=50;
static bool requireSecondaryMonitor=true;
static std::atomic<uint64_t> presentationCount=0;
static uint64_t lastPresentationMs=0;
static fs::path captureDir;
static DWORD simulationThread=0;
struct Vec3 { float x,y,z; };
static Vec3 previousPosition{};
static float measuredSpeed=0;
static float previousHealth=0,measuredDamage=0;
struct ShootParams {float distance,radius,accuracyAdd,critAdd,critMul,rpsAdd,customRPS,suppressionScale; int coverReduction; bool ignoreArmor,noSound; char pad[2];};
static_assert(sizeof(ShootParams)==40);
template<class T> T& mem(uintptr_t p,size_t o=0){return *reinterpret_cast<T*>(p+o);}
template<class F> F fn(uintptr_t r){return reinterpret_cast<F>(base+r);}
static bool valid(uintptr_t p,size_t bytes){MEMORY_BASIC_INFORMATION m{}; if(!p||!VirtualQuery((void*)p,&m,sizeof(m)))return false; return m.State==MEM_COMMIT&&!(m.Protect&(PAGE_NOACCESS|PAGE_GUARD))&&p+bytes<=reinterpret_cast<uintptr_t>(m.BaseAddress)+m.RegionSize;}
static void log(const std::string&s){logfile<<GetTickCount64()<<" "<<s<<std::endl;}
static json vec(Vec3 v){return json::array({v.x,v.y,v.z});}
static Vec3 parseVec(const json&v){if(!v.is_array()||v.size()!=3)throw std::runtime_error("expected vector3"); Vec3 x{v[0],v[1],v[2]};if(!std::isfinite(x.x)||!std::isfinite(x.y)||!std::isfinite(x.z))throw std::runtime_error("nonfinite vector");return x;}
static std::vector<uintptr_t> humans(uintptr_t server){
 std::vector<uintptr_t> out;if(!valid(server,639176))return out;
 uintptr_t head=server+516672,node=mem<uintptr_t>(head,8);
 for(int k=0;node&&node!=head&&k<20000;k++){
  if(!valid(node,32))break;auto owner=mem<uintptr_t>(node,24);
  if(valid(owner,5464)&&mem<uintptr_t>(owner)==base+rva::HumanVtable)out.push_back(owner);
  auto next=mem<uintptr_t>(node,8);if(next==node)break;node=next;
 }return out;
}
static uintptr_t equipped(uintptr_t h){if(!valid(h,5464))return 0;auto p=mem<uintptr_t>(h,5016);return valid(p,64)?p:0;}
static json humanInfo(uintptr_t h){
 auto e=equipped(h);int ammo=-1,cap=-1,state=-1;
 // InitBullets and AI empty checks confirm low 16 bits = bullets, high = projectiles.
 if(e){state=mem<int>(e,44);auto t=mem<uintptr_t>(e,32);if(valid(t,264)&&mem<int>(t,32)==0&&valid(e,104)){cap=mem<unsigned short>(e,74)+(mem<bool>(e,70)?1:0);ammo=mem<unsigned short>(e,60);}}
 return {{"id",mem<uint32_t>(h,16)},{"kind",mem<int>(h,312)},{"position",vec(mem<Vec3>(h,108))},{"aim",vec(mem<Vec3>(h,576))},{"look",vec(mem<Vec3>(h,536))},{"health",mem<float>(h,40)},{"ammo",ammo},{"capacity",cap},{"weapon_state",state},{"crouched",bool(mem<uint32_t>(h,508)&32u)},{"speed",mem<uint32_t>(h,16)==operatorId?measuredSpeed:0.f},{"recent_damage",mem<uint32_t>(h,16)==operatorId?measuredDamage:0.f},{"commands",mem<uint32_t>(h,4632)-mem<uint32_t>(h,4636)}};
}
#include "learning_bridge.h"
static json observation(uintptr_t server){
 json j={{"ok",true},{"protocol",1},{"episode",episode},{"sequence",sequence},{"active",bool(active)},{"simulation_thread",simulationThread},{"shot_requests",shotRequests},{"shots_accepted",shotsAccepted},{"automatic_thinks_blocked",blockedAuto},{"reload_requests",reloadRequests},{"sim_time_ms",valid(server,12)?mem<int>(server,8):0},{"game_state",valid(server,660)?mem<int>(server,656):0},{"humans",json::array()},{"visible_enemies",json::array()}};
 auto hs=humans(server);selected=0;
 for(auto h:hs){j["humans"].push_back(humanInfo(h));if(mem<uint32_t>(h,16)==operatorId)selected=h;}
 if(selected){j["operator"]=humanInfo(selected);int n=mem<int>(selected,5288+16);uintptr_t list=mem<uintptr_t>(selected,5288+8);j["visible_ids"]=json::array();
  if(n>=0&&n<20000&&valid(list,size_t(n)*4))for(int i=0;i<n;i++){uint32_t id=mem<uint32_t>(list,i*4);j["visible_ids"].push_back(id);for(auto h:hs)if(mem<uint32_t>(h,16)==id&&mem<int>(h,312)==1)j["visible_enemies"].push_back(humanInfo(h));}
 }j["presentation_count"]=presentationCount.load();j["presentation_interval_ms"]=presentIntervalMs;
 auto client=mem<uintptr_t>(base+rva::ClientGlobal);
 if(valid(client,0x7e214))j["client_view"]={{"sim_time_ms",mem<int>(client,8)},{"game_flags",mem<uint32_t>(client,660)},{"state",mem<int>(client,0x7e208)}};
 j["deployment_pending"]=deployRequested.load();
 addLearningObservation(j,server);return j;
}
using UpdateFn=void(__fastcall*)(uintptr_t,uint64_t);
using PauseFn=bool(__fastcall*)(uintptr_t);
using ThinkFn=void(__fastcall*)(uintptr_t);
static UpdateFn originalUpdate;
static UpdateFn originalClientUpdate;
static PauseFn originalPause;
static ThinkFn originalThink;
static void __fastcall clientUpdate(uintptr_t client,uint64_t realUs){
 if(loadRequested.exchange(false)){
  std::string path=requestedMap;
  int index=fn<int(__fastcall*)(uintptr_t,const char*)>(rva::MapIndex)(client,path.c_str());
  if(index<0){path=(root/"work/profile/KillHouseGames/DoorKickers2").string()+"/"+requestedMap;index=fn<int(__fastcall*)(uintptr_t,const char*)>(rva::MapIndex)(client,path.c_str());}
  log("validation map index "+std::to_string(index));
  if(index>=0){mem<int>(client,0x7e214)=index;restartRequested=true;}
 }
 if(restartRequested.exchange(false))fn<void(__fastcall*)(uintptr_t)>(rva::ClientRestart)(client);
 originalClientUpdate(client,realUs);
 if(deployRequested.load()&&!(mem<uint32_t>(client,660)&0x200)){
  // Fixed-start laboratory maps bypass the troop-selection UI. The server's
  // deployment completes, but the stock client can retain that panel. Finish
  // its native lifecycle on the client thread, only while the panel is open.
  auto gui=mem<uintptr_t>(base+0xfe71a8);
  if(valid(gui,0x1fc)&&mem<int>(gui,0x1f8)!=0)fn<void(__fastcall*)(uintptr_t)>(rva::ClientDeployFinished)(client);
  deployRequested=false;
 }
}
static bool __fastcall paused(uintptr_t s){return active ? !stepping : originalPause(s);}
static void __fastcall think(uintptr_t brain){
 auto owner=valid(brain,16)?mem<uintptr_t>(brain,8):0;
 if(active&&valid(owner,20)&&mem<uint32_t>(owner,16)==operatorId){++blockedAuto;return;}originalThink(brain);
}
static void perform(const json&a){
 if(!selected||mem<float>(selected,40)<=0)throw std::runtime_error("operator unavailable");
 auto action=a.value("action",std::string("stop"));auto e=equipped(selected);
 if(performLearningAction(a,mem<uintptr_t>(base+rva::ServerGlobal)))return;
 if(action=="move"){
  Vec3 dest=parseVec(a.at("destination")); Vec3 origin=mem<Vec3>(selected,108);Vec3 pts[2]={origin,dest};
  auto d=dest;d.x-=origin.x;d.z-=origin.z;if(d.x*d.x+d.z*d.z>400.)throw std::runtime_error("movement exceeds 20m local action limit");
  fn<void(__fastcall*)(uintptr_t,const Vec3*,int,const void*)>(rva::Waypoints)(selected,pts,2,nullptr);mem<uint32_t>(selected,508)&=~1u;
 }else if(action=="stop"){
  Vec3 here=mem<Vec3>(selected,108);
  fn<void(__fastcall*)(uintptr_t,const Vec3*,int,const void*)>(rva::Waypoints)(selected,&here,1,nullptr);
  mem<uint32_t>(selected,508)|=1u;
 }
 else if(action=="aim"||action=="turn_left"||action=="turn_right"||action=="fire"){
  Vec3 dir=a.contains("direction")?parseVec(a["direction"]):mem<Vec3>(selected,576);
  if(dir.x*dir.x+dir.z*dir.z<1e-8)throw std::runtime_error("zero direction");
  float norm=std::sqrt(dir.x*dir.x+dir.z*dir.z);dir={dir.x/norm,0,dir.z/norm};
  if(action=="turn_left"||action=="turn_right"){
   // Same inlined look interpolation initialization used by BrainPlayer::Think.
   mem<uint32_t>(selected,508)=(mem<uint32_t>(selected,508)&~8u)|2u;
   mem<Vec3>(selected,524)=mem<Vec3>(selected,536);mem<Vec3>(selected,548)=dir;
  }
  mem<uint32_t>(selected,508)&=~16u;
  fn<void(__fastcall*)(uintptr_t,const Vec3&)>(rva::Aim)(selected,dir);
  if(e&&mem<int>(e,44)==3)fn<void(__fastcall*)(uintptr_t)>(rva::ReadyAim)(e);
  if(action=="fire"){
   ShootParams p{a.value("distance",10.f),.25f,0,0,1,0,0,1,0,false,false,{0,0}};
   ++shotRequests;
   // CmdShoot alone accepts other states. Match the AI ready-state gate so the
   // normal firearm update owns aim delay, cyclic rate and reload timing.
   if(e&&equipmentType(e)==0&&mem<int>(e,44)==5&&mem<unsigned short>(e,60)>0)
    if(fn<bool(__fastcall*)(uintptr_t,const ShootParams&)>(rva::Shoot)(selected,p))++shotsAccepted;
  }
 }else if(action=="reload"){
  // Queue the game's own reload callback, rather than synthesizing ammunition.
  if(!e||equipmentType(e)!=0||mem<int>(e,44)==10||mem<uint32_t>(selected,4632)!=mem<uint32_t>(selected,4636))throw std::runtime_error("reload unavailable: equipment or queue busy");
  auto c=fn<uintptr_t(__fastcall*)(uintptr_t,int,uintptr_t,uintptr_t)>(rva::CreateCommand)(selected,10,base+rva::ReloadCommand,0);
  if(c){mem<int>(c,4)=0;mem<int>(c,8)=1;++reloadRequests;}
 }else throw std::runtime_error("unknown action");
}
static void publish(json j){std::lock_guard<std::mutex> lock(mu);reply=std::move(j);responding=true;SetEvent(replyEvent);}
static void __fastcall update(uintptr_t server,uint64_t realUs){
 simulationThread=GetCurrentThreadId();json cmd;bool have=false;
 {std::lock_guard<std::mutex>lock(mu);if(pending){cmd=request;pending=false;have=true;}}
 bool doStep=false;int duration=0;
 try{
  if(have){
   std::string op=cmd.value("op",std::string("observe"));
   if(op=="start"){
    if(active)throw std::runtime_error("session already active");
    if(mem<int>(server,656)!=1||(mem<uint32_t>(server,660)&0x1e00))throw std::runtime_error("deploy a running local mission first");
    for(int i=1;i<4;i++)if(mem<int>(server,518692+i*30120)!=0)throw std::runtime_error("only a local single-player session is supported");
    operatorId=cmd.value("operator_id",0u);auto o=observation(server);
    if(operatorId==0){for(auto&h:o["humans"])if(h["kind"]==0&&h["capacity"].get<int>()>0){operatorId=h["id"];break;}o=observation(server);}
    if(!selected)throw std::runtime_error("no matching operator");
    previousPosition=mem<Vec3>(selected,108);previousHealth=mem<float>(selected,40);measuredSpeed=measuredDamage=0;
    episode++;sequence=0;active=true;clockSet=false;shotRequests=shotsAccepted=blockedAuto=reloadRequests=0;loadoutChoice=-1;
    actionReceipt=json::object();actionEvents=json::array();pendingAction=json::object();
    record=cmd.value("record",false);frameNo=0;
    if(record){captureDir=root/"work/frames"/(std::to_string(GetCurrentProcessId())+"_"+std::to_string(episode)+"_"+std::to_string(GetTickCount64()));fs::create_directories(captureDir);}
   }else if(op=="step"){
    if(!active||!connected)throw std::runtime_error("inactive session");
    if(cmd.value("episode",uint64_t(0))!=episode||cmd.value("sequence",uint64_t(0))!=sequence+1)throw std::runtime_error("stale episode or sequence");
    duration=cmd.value("dt_ms",33);if(duration<1||duration>50)throw std::runtime_error("dt_ms must be 1..50");
    if(captureRequested.load()!=0)throw std::runtime_error("previous frame capture still pending");
    observation(server);auto action=cmd.value("command",json::object());
    beginAction(action,server);
    try{perform(action);}catch(const std::exception&e){receiptStatus("rejected",e.what());actionEvents.push_back(actionReceipt);pendingAction=json::object();}
    sequence++;doStep=true;
   }else if(op=="pause"){}else if(op=="end"){active=false;selected=0;record=false;clockSet=false;}
   else if(op=="load_validation"||op=="restart"||op=="load_mission"){
    deployRequested=false;
    if(op=="load_mission"){
     std::string name=cmd.value("mission",std::string());
     if(name.empty()||name.size()>100||name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("invalid laboratory mission name");
     auto path=root/"work/profile/KillHouseGames/DoorKickers2/data/maps_wip"/(name+".xml");
     if(!fs::exists(path))throw std::runtime_error("laboratory mission does not exist");requestedMap="data/maps_wip/"+name+".xml";
    }else if(op=="load_validation")requestedMap="data/maps_wip/fly_validation.xml";
    scenarioResult=0;actionReceipt=json::object();actionEvents=json::array();pendingAction=json::object();
    active=false;record=false;clockSet=false;selected=0;
    if(mem<int>(server,518692)!=0){
     auto done=mem<HANDLE>(base+0xff24e0);if(WaitForSingleObject(done,5000)!=WAIT_OBJECT_0)throw std::runtime_error("visibility busy before reload");
     std::string path=requestedMap;std::vector<unsigned char> bytes(path.begin(),path.end());bytes.push_back(0);bytes.push_back(0);
     struct Packed {unsigned char*data;int size,byte,bits,pad;uint64_t cache;} packed{bytes.data(),int(bytes.size()),0,0,0,0};
     fn<void(__fastcall*)(uintptr_t,uintptr_t,int,int,const Packed&)>(rva::ClientCommand)(server,server+518688,0,15,packed);
    }else loadRequested=true;
   }
   else if(op=="deploy"){
    if(mem<uint32_t>(server,660)&0x200){
     unsigned char zero[8]{};
     struct Packed {unsigned char* data;int size,byte,bits,pad;uint64_t cache;} packed{zero,8,0,0,0,0};
     static_assert(sizeof(Packed)==32);
     fn<void(__fastcall*)(uintptr_t,uintptr_t,const Packed&)>(rva::DeployServer)(server,server+518688,packed);
     deployRequested=true;
    }
   }
   else if(op!="observe")throw std::runtime_error("unknown operation");
  }
  if(active){
   if(!clockSet){virtualUs=realUs;clockSet=true;}
   virtualUs+=uint64_t(doStep?duration:1)*1000;stepping=doStep;
   // Movement reads per-client pause flags directly, independently of IsGamePaused.
   auto pauseBits=mem<uint32_t>(server,660)&0xf0u;
   if(doStep)mem<uint32_t>(server,660)&=~0xf0u;
   originalUpdate(server,virtualUs);
   // Server Update schedules visibility work. Wait for its normal completion
   // event before taking a coherent observation or restoring the paused state.
   if(doStep){
    auto done=mem<HANDLE>(base+0xff24e0);
    auto result=WaitForSingleObject(done,5000);
    mem<uint32_t>(server,660)|=pauseBits;
    if(result!=WAIT_OBJECT_0)throw std::runtime_error("visibility completion timeout");
   }
   stepping=false;
   if(doStep&&selected){auto p=mem<Vec3>(selected,108);float x=p.x-previousPosition.x,z=p.z-previousPosition.z;measuredSpeed=std::sqrt(x*x+z*z)*1000.f/duration;previousPosition=p;float health=mem<float>(selected,40);measuredDamage=std::max(0.f,previousHealth-health);previousHealth=health;}
  }
  else originalUpdate(server,realUs);
  if(doStep)settleAction(server);
  if(have){auto out=observation(server);out["requested_dt_ms"]=duration;
   out["game_flags"]=mem<uint32_t>(server,660);out["client0_state"]=mem<int>(server,518692);
   out["map_load_id"]=mem<unsigned char>(server,516632);
   if(doStep&&record){captureTime=out["sim_time_ms"];out["capture_frame"]=frameNo.load();out["capture_directory"]=captureDir.string();captureRequested=1;}
   publish(out);
  }
 }catch(const std::exception&e){stepping=false;if(have)publish({{"ok",false},{"error",e.what()},{"episode",episode},{"sequence",sequence}});log(std::string("exception: ")+e.what());}
}
using SwapFn=BOOL(WINAPI*)(HDC);static SwapFn originalSwap;
static BOOL CALLBACK secondaryMonitor(HMONITOR mon,HDC,LPRECT,LPARAM context){
 MONITORINFO mi{};mi.cbSize=sizeof(mi);
 if(GetMonitorInfoW(mon,&mi)&&!(mi.dwFlags&MONITORINFOF_PRIMARY)){*reinterpret_cast<RECT*>(context)=mi.rcWork;return FALSE;}return TRUE;
}
static bool windowPlaced=false;
static BOOL WINAPI swap(HDC dc){
 // Limit wall-clock presentation only. The explicit simulation step remains unchanged.
 const auto now=GetTickCount64();
 if(lastPresentationMs&&now-lastPresentationMs<presentIntervalMs)Sleep(DWORD(presentIntervalMs-(now-lastPresentationMs)));
 if(!windowPlaced){
  HWND window=WindowFromDC(dc);RECT area{};
  if(requireSecondaryMonitor){
   EnumDisplayMonitors(nullptr,nullptr,secondaryMonitor,reinterpret_cast<LPARAM>(&area));
  }else if(window){
   MONITORINFO mi{};mi.cbSize=sizeof(mi);
   auto monitor=MonitorFromWindow(window,MONITOR_DEFAULTTOPRIMARY);
   if(monitor&&GetMonitorInfoW(monitor,&mi))area=mi.rcWork;
  }
  if(window&&area.right>area.left){
   int w=std::min(1280L,area.right-area.left-24),h=std::min(750L,area.bottom-area.top-24);
   SetWindowPos(window,nullptr,area.left+12,area.top+12,w,h,SWP_NOZORDER|SWP_NOACTIVATE);
   log(std::string(requireSecondaryMonitor?"secondary":"available")+" monitor placement "+std::to_string(area.left)+","+std::to_string(area.top)+" size "+std::to_string(w)+"x"+std::to_string(h));
   windowPlaced=true;
  }
 }
 if(captureRequested.load()==1&&wglGetCurrentContext()){
  RECT rect{};HWND hwnd=WindowFromDC(dc);GetClientRect(hwnd,&rect);int w=rect.right,h=rect.bottom;
  if(w>0&&h>0&&w<=7680&&h<=4320){
   std::vector<unsigned char> pixels(size_t(w)*h*3);GLint oldPack=4,oldRead=GL_BACK;glGetIntegerv(GL_PACK_ALIGNMENT,&oldPack);glGetIntegerv(GL_READ_BUFFER,&oldRead);glPixelStorei(GL_PACK_ALIGNMENT,1);glReadBuffer(GL_BACK);glReadPixels(0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,pixels.data());glPixelStorei(GL_PACK_ALIGNMENT,oldPack);glReadBuffer(oldRead);
   stbi_flip_vertically_on_write(1);char name[64];int currentFrame=frameNo.load();sprintf_s(name,"frame_%06d.png",currentFrame);auto p=captureDir/name;
   int ok=stbi_write_png(p.string().c_str(),w,h,3,pixels.data(),w*3);
   auto client=mem<uintptr_t>(base+rva::ClientGlobal);
   std::ofstream f(captureDir/"timestamps.jsonl",std::ios::app);f<<json({{"frame",currentFrame},{"sim_time_ms",captureTime},{"client_sim_time_ms",mem<int>(client,8)},{"width",w},{"height",h},{"saved",bool(ok)}}).dump()<<'\n';f.close();frameNo++;
  }
  captureRequested=0;
 }BOOL result=originalSwap(dc);lastPresentationMs=GetTickCount64();presentationCount++;return result;
}
static bool ioExact(HANDLE h,void*buffer,DWORD bytes,bool writing){auto p=(char*)buffer;DWORD n;while(bytes){BOOL ok=writing?WriteFile(h,p,bytes,&n,nullptr):ReadFile(h,p,bytes,&n,nullptr);if(!ok||!n)return false;p+=n;bytes-=n;}return true;}
static DWORD WINAPI pipeLoop(void*){
 std::wstring name=L"\\\\.\\pipe\\FlyOperator_"+std::to_wstring(GetCurrentProcessId());
 for(;;){
  HANDLE pipe=CreateNamedPipeW(name.c_str(),PIPE_ACCESS_DUPLEX|FILE_FLAG_FIRST_PIPE_INSTANCE,PIPE_TYPE_BYTE|PIPE_READMODE_BYTE|PIPE_WAIT|PIPE_REJECT_REMOTE_CLIENTS,1,65536,65536,0,nullptr);
  if(pipe==INVALID_HANDLE_VALUE){log("pipe creation failed "+std::to_string(GetLastError()));return 1;}
  if(ConnectNamedPipe(pipe,nullptr)||GetLastError()==ERROR_PIPE_CONNECTED){connected=true;
   for(;;){uint32_t n=0;if(!ioExact(pipe,&n,4,false)||n==0||n>65536)break;std::string data(n,'\0');if(!ioExact(pipe,data.data(),n,false))break;
    try{json cmd=json::parse(data);ResetEvent(replyEvent);{std::lock_guard<std::mutex>lock(mu);request=cmd;pending=true;responding=false;}
     if(WaitForSingleObject(replyEvent,30000)!=WAIT_OBJECT_0){log("simulation reply timeout");break;}
     std::string out;{std::lock_guard<std::mutex>lock(mu);out=reply.dump();}n=(uint32_t)out.size();if(!ioExact(pipe,&n,4,true)||!ioExact(pipe,out.data(),n,true))break;
    }catch(const std::exception&e){log(e.what());break;}
   }
  }connected=false;stepping=false;{std::lock_guard<std::mutex>lock(mu);pending=false;}DisconnectNamedPipe(pipe);CloseHandle(pipe);
 }
}
using FolderFn=HRESULT(WINAPI*)(HWND,int,HANDLE,DWORD,LPWSTR);static FolderFn originalFolder;
static HRESULT WINAPI folder(HWND hwnd,int csidl,HANDLE token,DWORD flags,LPWSTR dest){
 if((csidl&0xff)==CSIDL_LOCAL_APPDATA){auto p=(root/"work/profile").wstring();wcscpy_s(dest,MAX_PATH,p.c_str());return S_OK;}return originalFolder(hwnd,csidl,token,flags,dest);
}
static std::string sha256(const fs::path&p){
 BCRYPT_ALG_HANDLE alg;BCRYPT_HASH_HANDLE hash;BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0);BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0);
 std::ifstream f(p,std::ios::binary);char b[65536];while(f){f.read(b,sizeof(b));if(f.gcount())BCryptHashData(hash,(PUCHAR)b,(ULONG)f.gcount(),0);}unsigned char digest[32];BCryptFinishHash(hash,digest,32,0);BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);char result[65]{};for(int i=0;i<32;i++)sprintf_s(result+2*i,3,"%02x",digest[i]);return result;
}
static DWORD WINAPI init(void*mod){
 wchar_t path[32768];GetModuleFileNameW((HMODULE)mod,path,32768);root=fs::path(path).parent_path().parent_path().parent_path();logfile.open(root/"work/bridge.log",std::ios::app);base=(uintptr_t)GetModuleHandleW(nullptr);
 GetModuleFileNameW(nullptr,path,32768);if(sha256(path)!=EXPECTED_SHA256){log("REFUSED binary hash mismatch");return 1;}
 auto hook=[&](uintptr_t target,void*replacement,void**original){auto result=MH_CreateHook((void*)target,replacement,original);if(result!=MH_OK)throw std::runtime_error(MH_StatusToString(result));};
 try{
  std::ifstream config(root/"config/learning.json");if(config){json settings;config>>settings;presentIntervalMs=unsigned(std::max(16,std::min(1000,settings.value("game_present_interval_ms",50))));}
  std::ifstream runtime(root/"config/runtime.json");if(runtime){json settings;runtime>>settings;requireSecondaryMonitor=settings.value("require_secondary_monitor",true);}
  SetPriorityClass(GetCurrentProcess(),BELOW_NORMAL_PRIORITY_CLASS);
  if(MH_Initialize()!=MH_OK)throw std::runtime_error("MinHook init");
  hook((uintptr_t)&SHGetFolderPathW,(void*)folder,(void**)&originalFolder);
  hook(base+rva::ServerUpdate,(void*)update,(void**)&originalUpdate);hook(base+rva::ServerPaused,(void*)paused,(void**)&originalPause);hook(base+rva::Think,(void*)think,(void**)&originalThink);
  hook(base+rva::ClientUpdate,(void*)clientUpdate,(void**)&originalClientUpdate);
  hook(base+rva::ScenarioEvaluate,(void*)evaluateScenario,(void**)&originalScenario);
  hook((uintptr_t)&SwapBuffers,(void*)swap,(void**)&originalSwap);
  if(MH_EnableHook(MH_ALL_HOOKS)!=MH_OK)throw std::runtime_error("enable hooks");
  replyEvent=CreateEventW(nullptr,TRUE,FALSE,nullptr);CreateThread(nullptr,0,pipeLoop,nullptr,0,nullptr);
  log("READY build hash verified; hooks installed");
  auto name=L"Local\\FlyBridgeReady_"+std::to_wstring(GetCurrentProcessId());auto event=OpenEventW(EVENT_MODIFY_STATE,FALSE,name.c_str());if(event){SetEvent(event);CloseHandle(event);}
 }catch(const std::exception&e){log(std::string("INIT FAILED: ")+e.what());return 1;}
 return 0;
}
BOOL WINAPI DllMain(HINSTANCE h,DWORD reason,LPVOID){if(reason==DLL_PROCESS_ATTACH){DisableThreadLibraryCalls(h);HANDLE thread=CreateThread(nullptr,0,init,h,0,nullptr);if(thread)CloseHandle(thread);}return TRUE;}
