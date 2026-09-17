from pathlib import Path
import json
r=Path(__file__).resolve().parents[1];s=json.loads((r/'work/re/symbols.json').read_text())
names=['Firearm::Fire','Firearm::Reload','Firearm::DoReload','Firearm::ReadyAim','Firearm::InitBullets','Human_Server::CreateQueueCommand','Human_Server::ProcessCmdFirearmReload','Human_Server::CmdAimInDirection','GameClient::DoRestartMap','GameClient::LoadMap','AI::Human_TryCrouchedReloadingIfEmpty']
(r/'work/re/extra_targets.tsv').write_text('\n'.join(f'{0x140000000+s[n]["rva"]:x}\t{n}' for n in names if n in s))
