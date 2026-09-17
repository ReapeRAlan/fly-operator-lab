#pragma once
#include <cstdint>
namespace rva {
constexpr uintptr_t ServerUpdate=0x2c2d30; // GameServer::Update
constexpr uintptr_t ServerPaused=0x2c9030; // GameServer::IsGamePaused
constexpr uintptr_t Think=0x2831e0; // AI::BrainPlayer::Think
constexpr uintptr_t HumanInit=0x293ac0; // Human_Server::Init
constexpr uintptr_t Aim=0x29f610; // Human_Server::CmdAimInDirection
constexpr uintptr_t Shoot=0x2a10e0; // Human_Server::CmdFirearmShoot
constexpr uintptr_t Waypoints=0x299e30; // Human_Server::ReplaceWaypoints
constexpr uintptr_t ReadyAim=0x2bd3b0; // Firearm::ReadyAim
constexpr uintptr_t CreateCommand=0x29f910; // Human_Server::CreateQueueCommand
constexpr uintptr_t ReloadCommand=0x2a0f90; // Human_Server::ProcessCmdFirearmReload
constexpr uintptr_t ClientRestart=0x17f440; // GameClient::DoRestartMap
constexpr uintptr_t HumanVtable=0x74ba38; // Human_Server::`vftable'
constexpr uintptr_t ServerGlobal=0xff2838; // g_pGameServer
constexpr uintptr_t ClientGlobal=0xff0ea8; // g_pGameClient
constexpr uintptr_t ClientUpdate=0x182900; // GameClient::Update
constexpr uintptr_t MapIndex=0x17f650; // GameClient::GetMapIndexByFileName
constexpr uintptr_t TogglePlay=0x187930; // GameClient::TogglePlayPause
constexpr uintptr_t ClientDeployFinished=0x187200; // GameClient::OnDeployFinished
constexpr uintptr_t DeployServer=0x2c5750; // GameServer::Net_OnDeployFinishedCommand
constexpr uintptr_t ClientCommand=0x2c6090; // GameServer::Net_OnClientCommand
constexpr uintptr_t ItemEquip=0x2a1910; // Human_Server::ProcessCmdItemEquip
constexpr uintptr_t DoorOpen=0x2a2520; // Human_Server::CmdQueueDoorOpen
constexpr uintptr_t DoorBreach=0x2a5d80; // Human_Server::CmdQueueDoorBreach
constexpr uintptr_t GrenadeThrow=0x2a3b60; // Human_Server::CmdQueueGrenadeThrow
constexpr uintptr_t Use=0x2ab1e0; // Human_Server::CmdQueueUse
constexpr uintptr_t CancelUse=0x2ab4b0; // Human_Server::CmdCancelUse
constexpr uintptr_t Defuse=0x2aa230; // Human_Server::CmdQueueTimebombDefuse
constexpr uintptr_t Arrest=0x2a76a0; // Human_Server::CmdQueueArrest
constexpr uintptr_t Follow=0x2a8ee0; // Human_Server::CmdFollowToggle
constexpr uintptr_t Evacuate=0x2a9180; // Human_Server::CmdEvacuate
constexpr uintptr_t Crouch=0x29ff20; // Human_Server::CmdQueueCrouch
constexpr uintptr_t Interrupt=0x29fe90; // Human_Server::InterruptQueuedCommands
constexpr uintptr_t ClearObstacle=0x2a9b80; // Human_Server::CmdQueueClearObstacle
constexpr uintptr_t SpyCamera=0x2a6520; // Human_Server::CmdQueueSpyCamera
constexpr uintptr_t ScenarioEvaluate=0x2ca0b0; // Scenario_Server::Evaluate
constexpr uintptr_t DoorVtable=0x74a240; // Door_Server::`vftable'
constexpr uintptr_t NewEquipment=0x2bc070; // Equipment::New
constexpr uintptr_t InventoryEquip=0x2c03f0; // Inventory_Server::Equip
constexpr uintptr_t SetEquipment=0x2a11c0; // Human_Server::SetEquippedItem
constexpr uintptr_t CancelBreach=0x2a6000; // Human_Server::CmdCancelDoorBreach
}
constexpr char EXPECTED_SHA256[]="7f873f34dfe37dfe40840140baf42352ea8f6b8e144d652a261a3dbddf83d747";
constexpr char EXPECTED_PDB_GUID[]="a2347bb6-eca6-424f-8b5c-a6431e79860c";