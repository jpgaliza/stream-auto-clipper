# stream-auto-clipper

Pipeline 100% local e agnóstica de nicho: VODs de lives (YouTube, Twitch, Kick) ou `.mp4` local → Shorts/TikTok/Reels 1080x1920
(webcam 35% no topo, conteúdo 65% na base, divisória esmeralda e legendas karaokê).

## Índice
1. [Instalação (do git clone ao primeiro uso)](#1-instalação)
2. [Comandos](#2-comandos)
3. [Etapas do pipeline](#3-etapas-do-pipeline)
4. [Chat](#4-chat)
5. [Configurar YouTube e TikTok](#5-configurar-upload-automático)
6. [Limpeza](#6-limpeza)
7. [Notas e solução de problemas](#7-notas)

---

## 1. Instalação

Requisitos: Python 3.10–3.13, Git e [ffmpeg](https://ffmpeg.org). Funciona em **Windows, Linux e macOS**.

| Sistema | Aceleração | Observação |
|---------|-----------|------------|
| Windows / Linux + NVIDIA (8 GB VRAM bastam) | Whisper em CUDA + render com `h264_nvenc` | Configuração padrão |
| Linux sem NVIDIA | Whisper em CPU, render `libx264` | Use `--whisper-device cpu --whisper-compute-type int8` (mais lento) |
| macOS (Apple Silicon/Intel) | Whisper em CPU, render `libx264` | Mesmos flags de CPU acima; o Ollama usa a GPU Metal sozinho |

A fonte **Montserrat ExtraBold** já está em `fonts/Montserrat-ExtraBold.ttf` e é copiada automaticamente para cada render:
não precisa instalar nada no sistema.

### Windows (PowerShell)

```powershell
git clone https://github.com/jpgaliza/stream-auto-clipper.git stream-auto-clipper
cd stream-auto-clipper

python -m venv .venv
.\.venv\Scripts\Activate.ps1
# (erro de política de execução? Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # ajuste cu124 à sua CUDA
pip install -r requirements.txt
playwright install                       # só se for usar upload no TikTok

winget install Gyan.FFmpeg               # reabra o terminal depois
winget install Ollama.Ollama             # ou https://ollama.com/download
ollama pull qwen3.5:9b
ollama pull qwen2.5vl:7b                 # opcional: só se usar --verify
```

### Linux (Debian/Ubuntu; adapte o gerenciador de pacotes)

```bash
git clone https://github.com/jpgaliza/stream-auto-clipper.git stream-auto-clipper
cd stream-auto-clipper

sudo apt update && sudo apt install -y python3-venv python3-pip ffmpeg
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU NVIDIA (precisa do driver)
# sem NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
playwright install --with-deps           # só se for usar upload no TikTok

curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:9b
ollama pull qwen2.5vl:7b                 # opcional: só se usar --verify
```

### macOS (Homebrew)

```bash
git clone https://github.com/jpgaliza/stream-auto-clipper.git stream-auto-clipper
cd stream-auto-clipper

brew install python ffmpeg git
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install torch                        # build padrão do macOS (sem CUDA)
pip install -r requirements.txt
playwright install                       # só se for usar upload no TikTok

brew install --cask ollama               # ou https://ollama.com/download
ollama pull qwen3.5:9b
ollama pull qwen2.5vl:7b                 # opcional: só se usar --verify
```

Em Linux/macOS sem CUDA, rode sempre com `--whisper-device cpu --whisper-compute-type int8`
(ou altere `WHISPER_DEVICE` e `WHISPER_COMPUTE_TYPE` em `config.py`). No Mac, o `h264_nvenc` não existe e o
render usa `libx264` automaticamente.

### Verificar a instalação

```bash
python -c "import torch, faster_whisper, librosa, cv2, mediapipe, pysubs2; print('CUDA:', torch.cuda.is_available())"
ffmpeg -version
ollama list              # o serviço deve estar rodando em localhost:11434
python main.py --help
```

Links: [ffmpeg](https://ffmpeg.org/download.html) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · [Ollama](https://ollama.com) · [PyTorch](https://pytorch.org/get-started/locally/) · [Montserrat](https://fonts.google.com/specimen/Montserrat) · [tiktok-uploader](https://github.com/wkaisertexas/tiktok-uploader)

---

## 2. Comandos

Rode na raiz do projeto com o venv ativo (Windows: `.\.venv\Scripts\Activate.ps1`; Linux/macOS: `source .venv/bin/activate`).
Os exemplos usam `/` nos caminhos, que funciona nos três sistemas.
Sem GPU NVIDIA, acrescente `--whisper-device cpu --whisper-compute-type int8` ao `main.py`.

### Modo interativo (mais rápido)
Basta rodar sem `--url` que o terminal pede o link, tema e opções passo a passo:
```bash
python main.py
```

### Pipeline completo (CLI direta)
```bash
python main.py --url "URL_OU_ARQUIVO.mp4" --topic "tema" --max-clips 15
python main.py --url "URL" --cookies-from-browser chrome        # se o site bloquear o download
python main.py --url "URL" --upload-yt                          # + YouTube (private)
python main.py --url "URL" --upload-tiktok                      # + TikTok (only_you)
python main.py --url "URL" --upload-all                         # ambos, em sequência
python main.py --url "URL" --verify --upload-all                # valida hook/contexto com VLM antes de enviar
python main.py --url "https://kick.com/canal" --live-minutes 30  # captura 30 min ao vivo + chat em tempo real
python main.py --url "URL" --start-at 5                         # retoma da etapa N (1-6)
```

Flags do `main.py`: `--url`, `--topic`, `--max-clips` (padrão 15), `--cookies-from-browser`, `--upload-yt`, `--upload-tiktok`,
`--upload-all`, `--start-at`, `--workdir`, `--ollama-model`, `--whisper-model`, `--min-clip`, `--max-clip`,
`--tiktok-browser`, `--tiktok-headless`.
Cada etapa roda em subprocesso próprio (logs em `output/<slug>/logs/`); no fim há um resumo com tempos e status de envio.

### Etapas isoladas (a pasta é `output/<slug>`)
```bash
python 01_ingest.py "URL" --cookies-from-browser chrome --topic "tema"
python 02_radar.py output/<slug> --top-n 40 --reaction-regex "LUL|catJAM"
python 03_transcribe.py output/<slug>
python 04_curator.py output/<slug> --topic "tema"
python 05_vision_render.py output/<slug> --max-clips 15
python 06_publisher.py output/<slug> --upload-all
```

---

## 3. Etapas do pipeline

| # | Script | Entrada → Saída |
|---|--------|-----------------|
| - | `config.py` | Constantes (1080x1920, 35/65, modelos, 15–58 s, radar…) e args comuns |
| 1 | `01_ingest.py` | URL/.mp4 → `video.mp4` (até 1080p60), `audio.wav` (mono 16 kHz), `chat_normalized.json`, `meta.json` |
| 2 | `02_radar.py` | áudio (RMS 2 s) + chat (msgs/s × reações) → `candidates.json` (top 40 janelas, −45 s/+20 s, mescladas até 180 s) |
| 3 | `03_transcribe.py` | só as janelas candidatas → `candidates_transcribed.json` (palavra a palavra, pt); libera a VRAM |
| 4 | `04_curator.py` | Ollama (`format=json`) → `approved_clips.json` por `virality_score`, cortes ajustados a fronteiras de frase |
| 5 | `05_vision_render.py` | detecta webcam (MediaPipe, 15 frames) → `corte_XX.mp4/.txt/.ass`, `rendered_clips.json` |
| 5b | `05b_verify.py` | (opcional `--verify`) VLM `qwen2.5vl:7b` avalia hook/contexto e auto-corrige cortes tardios → `verified_clips.json` |
| 6 | `06_publisher.py` | YouTube (`private`) e/ou TikTok (`only_you`) a partir de `verified_clips.json` (ou `rendered_clips.json`) |
| - | `clean.py` | Limpeza de downloads, cortes e cache |

Saídas ficam em `output/<slug-da-live>/`. Whisper `large-v3` (fp16) é descarregado antes de o Ollama carregar o LLM.

---

## 4. Chat

O chat é baixado automaticamente e normalizado para `chat_normalized.json` (`[{"timestamp", "user", "message"}]`):

- **YouTube:** `yt-dlp --write-subs --sub-langs live_chat` (VODs de lives com replay disponível).
- **Twitch:** `chat_downloader` (VODs). A API da Twitch muda com frequência e pode falhar.
- **Kick:** tentativa via `chat_downloader`, sem suporte oficial; provavelmente sem chat.
- **`.mp4` local:** aceita um `<nome>.live_chat.json` ao lado do arquivo.
- Sem chat, o radar usa apenas o áudio e o pipeline continua.

---

## 5. Configurar upload automático

### YouTube (Data API v3)
1. Acesse <https://console.cloud.google.com>, crie um projeto e ative a **YouTube Data API v3**.
2. Em *Tela de consentimento OAuth*, escolha **Externo** e adicione seu e-mail como **usuário de teste**.
3. Em *Credenciais* → *Criar credenciais* → **ID do cliente OAuth** → tipo **App para computador**.
4. Baixe o JSON e salve como `client_secrets.json` na raiz do projeto.
5. No primeiro upload o navegador abre para autorizar; o `token.json` fica salvo para os próximos.

Vídeos saem com `privacyStatus="private"` (revise no YouTube Studio). Projetos de API não verificados são forçados a privado e a cota padrão (10.000 unidades/dia) comporta cerca de 6 uploads por dia.

### TikTok (tiktok-uploader)
1. Faça login em tiktok.com no Chrome.
2. Com uma extensão como "Get cookies.txt LOCALLY", exporte os cookies de tiktok.com em **formato Netscape**.
3. Salve como `tiktok_cookies.txt` na raiz do projeto.
4. Rode `playwright install` (uma vez).
5. O envio usa visibilidade `only_you` (só você vê) para revisão antes de publicar. Cookies expiram; reexporte quando falhar.

A automação usa a página web do TikTok, que muda com frequência: se parar de funcionar, rode `pip install -U tiktok-uploader`.

### Rodar depois de configurado
```bash
python main.py --url "URL" --topic "tema" --max-clips 10 --upload-all
python main.py --url "URL" --start-at 6 --upload-all     # só enviar cortes já renderizados
```
`uploads.json` evita reenviar o que já subiu; o resumo final mostra o status por plataforma.

---

## 6. Limpeza

```bash
python clean.py                                  # lista o espaço usado por live (não apaga nada)
python clean.py --downloads --dry-run            # simula
python clean.py --downloads --slug minha-live    # vídeo/áudio/chat baixados dessa live
python clean.py --clips                          # corte_XX.* e rendered_clips.json
python clean.py --cache                          # webcam.json, logs, .ass, fonts copiadas
python clean.py --all --slug minha-live          # pasta inteira dessa live
python clean.py --all --yes                      # tudo, sem confirmação
```
Pede confirmação (exceto com `--yes`), pula arquivos em uso e nunca toca em `client_secrets.json`, `token.json`, `tiktok_cookies.txt`, `models/` ou `fonts/` da raiz.

---

## 7. Notas

- Sem webcam detectada de forma consistente, o render usa enquadramento 9:16 centralizado (sem divisória; legenda em Y=1344).
- O recorte da webcam usa a proporção do painel (1080:672 ≈ 16:9) com 15% de zoom interno.
- `models/blaze_face_short_range.tflite` é baixado automaticamente na primeira detecção de webcam.
- Sem NVENC, o render cai para `libx264` automaticamente.
- Tudo é ajustável em `config.py` (resolução, proporção, modelos, durações, pesos do radar).
- **Nunca** versione `client_secrets.json`, `token.json` ou `tiktok_cookies.txt` (já estão no `.gitignore`).