from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
def main():
    symbols=json.loads((ROOT/'work/re/symbols.json').read_text())
    names={'ServerUpdate':'GameServer::Update','ServerPaused':'GameServer::IsGamePaused','Think':'AI::BrainPlayer::Think','HumanInit':'Human_Server::Init','Aim':'Human_Server::CmdAimInDirection','Shoot':'Human_Server::CmdFirearmShoot','Waypoints':'Human_Server::ReplaceWaypoints','ReadyAim':'Firearm::ReadyAim','CreateCommand':'Human_Server::CreateQueueCommand','ReloadCommand':'Human_Server::ProcessCmdFirearmReload','ClientRestart':'GameClient::DoRestartMap','HumanVtable':"Human_Server::`vftable'",'ServerGlobal':'g_pGameServer','ClientGlobal':'g_pGameClient'}
    profile=json.loads((ROOT/'config/build.json').read_text())
    names['ClientUpdate']='GameClient::Update'
    names['MapIndex']='GameClient::GetMapIndexByFileName'
    names['TogglePlay']='GameClient::TogglePlayPause'
    names['ClientDeployFinished']='GameClient::OnDeployFinished'
    names['DeployServer']='GameServer::Net_OnDeployFinishedCommand'
    names['ClientCommand']='GameServer::Net_OnClientCommand'
    names.update({'ItemEquip':'Human_Server::ProcessCmdItemEquip','DoorOpen':'Human_Server::CmdQueueDoorOpen',
      'DoorBreach':'Human_Server::CmdQueueDoorBreach','GrenadeThrow':'Human_Server::CmdQueueGrenadeThrow',
      'Use':'Human_Server::CmdQueueUse','CancelUse':'Human_Server::CmdCancelUse',
      'Defuse':'Human_Server::CmdQueueTimebombDefuse','Arrest':'Human_Server::CmdQueueArrest',
      'Follow':'Human_Server::CmdFollowToggle','Evacuate':'Human_Server::CmdEvacuate',
      'Crouch':'Human_Server::CmdQueueCrouch','Interrupt':'Human_Server::InterruptQueuedCommands',
      'ClearObstacle':'Human_Server::CmdQueueClearObstacle','SpyCamera':'Human_Server::CmdQueueSpyCamera',
      'ScenarioEvaluate':'Scenario_Server::Evaluate','DoorVtable':"Door_Server::`vftable'",
      'NewEquipment':'Equipment::New','InventoryEquip':'Inventory_Server::Equip','SetEquipment':'Human_Server::SetEquippedItem',
      'CancelBreach':'Human_Server::CmdCancelDoorBreach'})
    lines=['#pragma once','#include <cstdint>','namespace rva {']
    for key,name in names.items(): lines.append(f'constexpr uintptr_t {key}=0x{symbols[name]["rva"]:x}; // {name}')
    lines+=['}',f'constexpr char EXPECTED_SHA256[]="{profile["sha256"]}";',f'constexpr char EXPECTED_PDB_GUID[]="{profile["guid"]}";']
    (ROOT/'native/build_profile.h').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main()
