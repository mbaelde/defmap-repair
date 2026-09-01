"""Position du clip musdb18-7s dans la piste complete, pour les 50 pistes test.

Lance par locate_clips.sh, qui fait le decodage. Lit des paires de fichiers bruts
f32le mono deja decodes par ffmpeg cote hote (le conteneur uv n'a pas ffmpeg),
nommes <slug>.clip.raw et <slug>.full.raw dans le repertoire passe en argument, et
sans argument joue son self-check. Ecrit une ligne TSV par piste sur stdout :

    slug <TAB> offset_s <TAB> rho <TAB> residu_gain_libre <TAB> gain <TAB> duree_piste_s

L'offset retenu maximise le coefficient de correlation rho, ce qui est
l'estimateur du maximum de vraisemblance de l'alignement a gain libre. Le juge de
la localisation est rho et le residu apres correction du gain, PAS l'ecart brut
au clip : les clips de musdb18-7s sont re-normalises plus fort que la piste, donc
un ecart brut de 1.0 peut coexister avec un alignement parfait. Un rho sous 0.9
est marque SUSPECT.

Correlation par FFT et non par np.correlate direct : une piste de 4 min contre un
clip de 7 s a 8 kHz coute 1e11 produits en direct, quelques dizaines de ms en FFT.
"""

import sys
from pathlib import Path

import numpy as np

SR = 8000
RHO_MIN = 0.9


def xcorr_valid(full: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """np.correlate(full, clip, 'valid'), calcule par FFT."""
    n, m = len(full), len(clip)
    size = 1 << (n + m - 1).bit_length()
    prod = np.fft.rfft(full, size) * np.fft.rfft(clip[::-1], size)
    return np.fft.irfft(prod, size)[m - 1:n]


def locate(clip: np.ndarray, full: np.ndarray) -> tuple[float, float, float, float]:
    """(offset en secondes, rho, residu apres gain libre, gain de la fenetre)."""
    m = len(clip)
    # energie glissante des fenetres de la piste, par sommes cumulees
    cum = np.concatenate(([0.0], np.cumsum(full.astype(np.float64) ** 2)))
    energy = cum[m:] - cum[:-m]
    num = xcorr_valid(full, clip)
    nclip = max(float(np.linalg.norm(clip)), 1e-12)
    rho = num / (np.sqrt(np.maximum(energy, 1e-12)) * nclip)
    # une fenetre vide (silence de fin de piste) donne un rho numeriquement
    # explosif, au-dela de 1 : elle ne peut pas etre le clip, on l'ecarte
    rho[energy < 1e-6 * nclip ** 2] = -np.inf
    best = int(np.argmax(rho))
    win = full[best:best + m]
    # meilleur gain a tel que a*clip approche win, et residu relatif restant
    gain = float(num[best]) / (nclip ** 2)
    nwin = max(float(np.linalg.norm(win)), 1e-12)
    res = float(np.linalg.norm(win - gain * clip) / nwin)
    return best / SR, float(rho[best]), res, gain


def demo() -> None:
    """Offset connu retrouve malgre un gain, y compris avec du silence en fin."""
    rng = np.random.default_rng(0)
    full = rng.standard_normal(SR * 30).astype(np.float32)
    full[SR * 25:] = 0.0  # le silence de fin est le piege qui faisait diverger
    for off, g in ((0.0, 1.0), (7.5, 0.7), (18.4, 3.0)):
        clip = full[int(off * SR):int(off * SR) + int(6.8 * SR)] * g
        got, rho, res, gain = locate(clip, full)
        assert abs(got - off) < 1e-6, (off, got)
        assert rho > 0.999 and res < 1e-5, (off, rho, res)
        assert abs(gain - 1.0 / g) < 1e-3, (g, gain)
    # bruit absent de la piste : rho doit rester sous le seuil, et jamais au-dela
    # de 1, ce qui etait le symptome de l'accrochage sur le silence de fin
    _, rho, _, _ = locate(rng.standard_normal(SR * 5).astype(np.float32), full)
    assert RHO_MIN > rho, rho
    print("demo ok: offsets retrouves a gain libre, silence de fin non piegeuse")


def main(raw_dir: Path) -> None:
    for clip_path in sorted(raw_dir.glob("*.clip.raw")):
        slug = clip_path.name[:-len(".clip.raw")]
        full_path = raw_dir / (slug + ".full.raw")
        if not full_path.exists():
            print(slug + "\tMANQUE\t\t", flush=True)
            continue
        clip = np.fromfile(clip_path, dtype=np.float32)
        full = np.fromfile(full_path, dtype=np.float32)
        off, rho, res, gain = locate(clip, full)
        flag = "\tSUSPECT" if rho < RHO_MIN else ""
        print("%s\t%.2f\t%.4f\t%.4f\t%.3f\t%.1f%s"
              % (slug, off, rho, res, gain, len(full) / SR, flag), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        demo()
    else:
        main(Path(sys.argv[1]))
