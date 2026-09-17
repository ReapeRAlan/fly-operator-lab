// Protocol v2 additions. PDB x64 layouts and decompiled routines in work/re.
static std::string requestedMap="data/maps_wip/fly_validation.xml";
static json actionReceipt=json::object(),actionEvents=json::array();
static json pendingAction=json::object();
static int scenarioResult=0;
static int loadoutChoice=-1;
using ScenarioFn=int(__fastcall*)(const char**,Vec3&,const char**);
static ScenarioFn originalScenario;
static int __fastcall evaluateScenario(const char**message,Vec3&position,const char**reason){
 int result=originalScenario(message,position,reason);scenarioResult=result;return result;
}
static std::string safeString(uintptr_t p,size_t maximum=128){
 std::string out;for(size_t i=0;i<maximum&&valid(p+i,1);i++){char c=mem<char>(p+i);if(!c)break;out.push_back(c);}return out;
}
static int equipmentType(uintptr_t e){if(!valid(e,64))return -1;auto t=mem<uintptr_t>(e,32);return valid(t,264)?mem<int>(t,32):-1;}
static json equipmentInfo(uintptr_t e,int slot){
 auto t=mem<uintptr_t>(e,32);int type=equipmentType(e);
 json j={{"slot",slot},{"type",type},{"name",valid(t,264)?safeString(mem<uintptr_t>(t,40)):""},
  {"state",mem<int>(e,44)},{"state_timer_ms",mem<int>(e,52)},{"state_duration_ms",mem<int>(e,56)},
  {"quantity",mem<int>(e,60)},{"equipped",selected&&mem<uintptr_t>(selected,5016)==e},{"effective_params",json::object()}};
 if(type==0&&valid(e,104)){j["ammo"]=mem<unsigned short>(e,60);j["capacity"]=mem<unsigned short>(e,74)+(mem<bool>(e,70)?1:0);}
 if(valid(t,264)){j["category"]=safeString(mem<uintptr_t>(t,96));j["binding"]=safeString(mem<uintptr_t>(t,112));}
 // ModifiableFloatVariableCollection: 40-byte entries: char name[32], hash, value.
 int n=mem<int>(e,24);auto vars=mem<uintptr_t>(e,16);
 if(n>=0&&n<=2048&&valid(vars,size_t(n)*40))for(int i=0;i<n;i++){
  float v=mem<float>(vars+i*40,36);if(std::isfinite(v))j["effective_params"][safeString(vars+i*40,32)]=v;
 }
 return j;
}
static std::vector<uintptr_t> allEntities(uintptr_t server){
 std::vector<uintptr_t> out;uintptr_t head=server+516672,node=mem<uintptr_t>(head,8);
 for(int k=0;node&&node!=head&&k<20000;k++){
  if(!valid(node,32))break;auto p=mem<uintptr_t>(node,24);if(valid(p,280))out.push_back(p);
  auto next=mem<uintptr_t>(node,8);if(next==node)break;node=next;
 }return out;
}
static uintptr_t entityById(uintptr_t server,uint32_t id){for(auto p:allEntities(server))if(mem<uint32_t>(p,16)==id)return p;return 0;}
static void receiptStatus(const std::string&status,const std::string&reason=""){
 actionReceipt["status"]=status;actionReceipt["reason"]=reason;
}
static float distanceXZ(Vec3 a,Vec3 b){float x=a.x-b.x,z=a.z-b.z;return std::sqrt(x*x+z*z);}
static void addLearningObservation(json&j,uintptr_t server){
 j["protocol"]=2;j["mission_result"]=scenarioResult;j["map_path"]=requestedMap;j["loadout_choice"]=loadoutChoice;
 j["action_receipt"]=actionReceipt;j["action_events"]=actionEvents;
 j["objects"]=json::array();j["inventory"]=json::array();j["map_seed"]=mem<uint32_t>(server,516628);
 if(!selected)return;
 for(int slot=1;slot<27;slot++){auto e=mem<uintptr_t>(selected,4800+slot*8);if(valid(e,64))j["inventory"].push_back(equipmentInfo(e,slot));}
 for(auto p:allEntities(server)){
  if(mem<uintptr_t>(p)==base+rva::HumanVtable)continue;
  auto t=mem<uintptr_t>(p,32);if(!valid(t,32))continue;
  auto name=safeString(mem<uintptr_t>(t,8));bool door=mem<uintptr_t>(p)==base+rva::DoorVtable;
  auto lower=name;for(char&c:lower)if(c>='A'&&c<='Z')c=char(c-'A'+'a');
  bool relevant=door||lower.find("bomb")!=std::string::npos||lower.find("rescue")!=std::string::npos||lower.find("extraction")!=std::string::npos||lower.find("use")!=std::string::npos||lower.find("obstacle")!=std::string::npos;
  if(!relevant)continue;
  json o={{"id",mem<uint32_t>(p,16)},{"template",name},{"position",vec(mem<Vec3>(p,108))},{"kind",door?"door":"object"},{"source","scenario_geometry"}};
  if(door&&valid(p,520)){o["door_state"]=mem<int>(p,312);o["locked"]=mem<bool>(p,329);o["initial_position"]=vec(mem<Vec3>(p,384));}
  j["objects"].push_back(o);
 }
}
static bool performLearningAction(const json&a,uintptr_t server){
 auto action=a.value("action",std::string("wait"));
 if(action=="wait")return true;
 if(!selected)throw std::runtime_error("operator unavailable");
 if(action=="loadout"){
  if(mem<int>(server,8)!=0||loadoutChoice!=-1)throw std::runtime_error("loadout selection is only available before the first simulation step");
  int kit=a.value("kit",-1);if(kit<0||kit>2)throw std::runtime_error("unsupported validated kit");
  // Curated native-compatible bundles. Their normal templates own all timings/modifiers.
  if(kit>0){
   struct Part{int slot;const char*name;};
   std::vector<Part> parts={{1,"M4 Carbine"},{2,"IronSights"},{3,"556FMJM855_M4"}};
   if(kit==2)parts.push_back({14,"Dynamic Hammer"});
   std::vector<uintptr_t> created;for(auto p:parts){auto item=fn<uintptr_t(__fastcall*)(const char*)>(rva::NewEquipment)(p.name);if(!valid(item,64))throw std::runtime_error("equipment template construction failed");created.push_back(item);}
   fn<void(__fastcall*)(uintptr_t,int)>(rva::SetEquipment)(selected,0);
   for(size_t k=0;k<parts.size();k++)fn<void(__fastcall*)(uintptr_t,uintptr_t,int,uintptr_t)>(rva::InventoryEquip)(selected+4800,created[k],parts[k].slot,selected);
   auto c=fn<uintptr_t(__fastcall*)(uintptr_t,int,uintptr_t,uintptr_t)>(rva::CreateCommand)(selected,0,base+rva::ItemEquip,0);
   if(!c)throw std::runtime_error("loadout equip queue allocation failed");mem<int>(c,88)=1;
  }
  loadoutChoice=kit;return true;
 }
 auto qcount=mem<uint32_t>(selected,4632)-mem<uint32_t>(selected,4636);
 int slot=a.value("slot",0);auto e=slot>0&&slot<27?mem<uintptr_t>(selected,4800+slot*8):0;
 auto target=entityById(server,a.value("target_id",0u));
 auto requireTarget=[&](){if(!target)throw std::runtime_error("target unavailable");if(distanceXZ(mem<Vec3>(selected,108),mem<Vec3>(target,108))>3.f)throw std::runtime_error("target outside 3m interaction radius");};
 auto requireItem=[&](){if(!valid(e,64))throw std::runtime_error("inventory slot empty");};
 if(action=="cancel"){
  bool breach=false;auto tail=mem<uint32_t>(selected,4636);
  for(uint32_t k=0;k<qcount&&k<32;k++)if(mem<int>(selected+792+((tail+k)&31)*120)==12)breach=true;
  if(!breach)throw std::runtime_error("no validated cancellable interaction; weapon cycles keep native timing");
  fn<void(__fastcall*)(uintptr_t)>(rva::CancelBreach)(selected);
  fn<void(__fastcall*)(uintptr_t)>(rva::Interrupt)(selected);return true;
 }
 bool advanced=action=="equip"||action=="door_open"||action=="door_breach"||action=="throw"||action=="use"||action=="defuse"||action=="arrest"||action=="follow"||action=="evacuate"||action=="crouch"||action=="clear_obstacle"||action=="spy_camera";
 if(!advanced)return false;
 if(qcount>0)throw std::runtime_error("operator command queue busy");
 if(action=="equip"){
  requireItem();if(slot!=1&&slot!=5)throw std::runtime_error("manual equip requires primary or secondary weapon");
  if(e==mem<uintptr_t>(selected,5016)){receiptStatus("completed","already_equipped");return true;}
  auto c=fn<uintptr_t(__fastcall*)(uintptr_t,int,uintptr_t,uintptr_t)>(rva::CreateCommand)(selected,0,base+rva::ItemEquip,0);
  if(!c)throw std::runtime_error("command queue allocation failed");mem<int>(c,88)=slot;
 }else if(action=="door_open"){
  requireTarget();if(mem<uintptr_t>(target)!=base+rva::DoorVtable)throw std::runtime_error("not a door");
  fn<void(__fastcall*)(uintptr_t,uintptr_t,bool)>(rva::DoorOpen)(selected,target,a.value("kick",false));
 }else if(action=="door_breach"){
  requireTarget();requireItem();if(mem<uintptr_t>(target)!=base+rva::DoorVtable)throw std::runtime_error("not a door");
  fn<void(__fastcall*)(uintptr_t,int,uintptr_t)>(rva::DoorBreach)(selected,slot,target);
 }else if(action=="throw"){
  requireItem();if(equipmentType(e)!=3||mem<int>(e,60)<=0)throw std::runtime_error("no grenade in this slot");
  Vec3 dest=parseVec(a.at("destination")),origin=mem<Vec3>(selected,108),zero{0,0,0};
  if(distanceXZ(origin,dest)>20.f)throw std::runtime_error("throw exceeds 20m action radius");
  fn<void(__fastcall*)(uintptr_t,int,const Vec3&,float,const Vec3&,uintptr_t)>(rva::GrenadeThrow)(selected,slot,dest,0.f,zero,0);
 }else if(action=="use"||action=="defuse"||action=="clear_obstacle"){
  requireTarget();auto r=action=="use"?rva::Use:action=="defuse"?rva::Defuse:rva::ClearObstacle;
  fn<void(__fastcall*)(uintptr_t,uint32_t)>(r)(selected,mem<uint32_t>(target,16));
 }else if(action=="arrest"||action=="follow"){
  requireTarget();if(mem<uintptr_t>(target)!=base+rva::HumanVtable)throw std::runtime_error("target not human");
  if(action=="arrest")fn<void(__fastcall*)(uintptr_t,uintptr_t,bool)>(rva::Arrest)(selected,target,true);
  else fn<void(__fastcall*)(uintptr_t,uintptr_t,bool)>(rva::Follow)(target,selected,true);
 }else if(action=="evacuate")fn<void(__fastcall*)(uintptr_t)>(rva::Evacuate)(selected);
 else if(action=="crouch")fn<void(__fastcall*)(uintptr_t,bool)>(rva::Crouch)(selected,a.value("value",true));
 else if(action=="spy_camera"){
  requireTarget();requireItem();Vec3 direction=parseVec(a.at("direction"));
  fn<void(__fastcall*)(uintptr_t,int,uint32_t,const Vec3&)>(rva::SpyCamera)(selected,slot,mem<uint32_t>(target,16),direction);
 }
 if(mem<uint32_t>(selected,4632)==mem<uint32_t>(selected,4636)&&action!="follow"&&action!="evacuate"&&action!="crouch")throw std::runtime_error("engine did not enqueue action");
 return true;
}
static void beginAction(const json&a,uintptr_t server){
 if(a.value("action",std::string("wait"))=="wait")return;
 if(!selected)throw std::runtime_error("operator unavailable");
 if(!pendingAction.empty()){auto old=actionReceipt;old["status"]="superseded";old["reason"]="new_command";actionEvents.push_back(old);}
 actionReceipt={{"command_sequence",sequence+1},{"action",a.value("action",std::string("stop"))},{"status","accepted"},{"reason",""},{"start_sim_ms",mem<int>(server,8)}};
 pendingAction=a;pendingAction["start_ammo"]=equipped(selected)?mem<unsigned short>(equipped(selected),60):-1;
 pendingAction["start_shots"]=shotsAccepted;pendingAction["start_sim_ms"]=mem<int>(server,8);
 int slot=a.value("slot",0);auto e=slot>0&&slot<27?mem<uintptr_t>(selected,4800+slot*8):0;
 pendingAction["start_quantity"]=valid(e,64)?mem<int>(e,60):-1;
}
static void settleAction(uintptr_t server){
 if(pendingAction.empty()||!selected)return;
 std::string action=pendingAction.value("action",std::string());bool done=false;
 int elapsed=mem<int>(server,8)-pendingAction.value("start_sim_ms",0);auto e=equipped(selected);
 if(actionReceipt.value("status",std::string())=="accepted")receiptStatus("in_progress");
 if(action=="fire"){
  if(shotsAccepted>pendingAction.value("start_shots",0))done=true;else receiptStatus("rejected","weapon_not_ready_or_empty");
 }else if(action=="stop"||action=="turn_left"||action=="turn_right"||action=="aim")done=true;
 else if(action=="cancel")done=mem<uint32_t>(selected,4632)==mem<uint32_t>(selected,4636);
 else if(action=="move")done=distanceXZ(mem<Vec3>(selected,108),parseVec(pendingAction.at("destination")))<.2f;
 else if(action=="reload")done=e&&mem<int>(e,44)!=10&&mem<unsigned short>(e,60)>pendingAction.value("start_ammo",0);
 else if(action=="equip"){
  int slot=pendingAction.value("slot",0);done=e&&slot>0&&slot<27&&e==mem<uintptr_t>(selected,4800+slot*8)&&mem<int>(e,44)>2;
 }else if(action=="throw"){
  int slot=pendingAction.value("slot",0);auto item=slot>0&&slot<27?mem<uintptr_t>(selected,4800+slot*8):0;
  done=valid(item,64)&&mem<int>(item,60)<pendingAction.value("start_quantity",0);
 }else if(action=="door_open"||action=="door_breach"){
  auto target=entityById(server,pendingAction.value("target_id",0u));done=target&&valid(target,520)&&(mem<int>(target,312)==0);
 }else if(action=="loadout")done=loadoutChoice==pendingAction.value("kit",-1)&&e&&equipmentType(e)==0&&mem<int>(e,44)>2;
 else if(action=="crouch")done=bool(mem<uint32_t>(selected,508)&32u)==pendingAction.value("value",true);
 else if(action=="defuse")done=scenarioResult==1; // single-bomb laboratory exercise; completion is a native outcome
 else if(action=="follow"){
  auto target=entityById(server,pendingAction.value("target_id",0u));auto brain=target?mem<uintptr_t>(target,5312):0;
  done=valid(brain,400)&&mem<int>(brain,16)==1&&mem<uint32_t>(brain,356)==operatorId;
 }
 else done=elapsed>0&&mem<uint32_t>(selected,4632)==mem<uint32_t>(selected,4636);
 if(done)receiptStatus("completed");
 if(elapsed>20000&&!done)receiptStatus("failed","no_confirmed_effect_within_20s");
 actionReceipt["elapsed_ms"]=elapsed;
 auto status=actionReceipt.value("status",std::string());
 if(status=="completed"||status=="rejected"||status=="failed"){actionEvents.push_back(actionReceipt);pendingAction=json::object();}
 while(actionEvents.size()>12)actionEvents.erase(actionEvents.begin());
}
