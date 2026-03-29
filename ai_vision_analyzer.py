# ============================================================
# ai_vision_analyzer.py - AI Vision Tabanlı Yol Analizi
# TUA Astro Hackathon DSS — Claude Vision API Entegrasyonu
# ============================================================

"""
Uydu görüntüsünü Claude Vision API ile analiz eder.

Çalışma mantığı:
  1. Görüntü → base64 encode
  2. Sabit prompt şablonu ile Claude'a gönder
  3. Yanıtı parse et → RoadStatus listesi
  4. Mevcut CV (SSIM) pipeline sonuçlarıyla karşılaştır

Bu modül internet + Anthropic API key gerektirir.
API key: .streamlit/secrets.toml → [anthropic] api_key
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from loguru import logger


# ── Sabitler ────────────────────────────────────────────────

ANALYSIS_PROMPT = """Ekteki 0.5m çözünürlüklü uydu görüntüsünde yer alan yol ağını analiz et.

Analizini şu üç kategoriye ayırarak JSON formatında sun:

1. Açık Yollar: Üzerinde trafik akışı olan veya hiçbir fiziksel engel bulunmayan pürüzsüz yollar.
2. Kapalı Yollar: İnşaat, bariyer, moloz veya yol çalışması nedeniyle fiziksel olarak geçişin imkansız olduğu yollar.
3. Kısıtlı Yollar: Hasarlı, daralmış veya sadece tek şeridi açık olan yollar.

Görüntüdeki görsel kanıtları (gölge boyları, iş makinesi varlığı, zemin dokusu değişikliği) belirterek hangi bölgedeki yolun neden bu kategoride olduğunu açıkla.

Yanıtını YALNIZCA şu JSON şemasında ver (başka metin ekleme):
{
  "summary": {
    "open_count": <int>,
    "closed_count": <int>,
    "restricted_count": <int>,
    "overall_assessment": "<genel değerlendirme metni>"
  },
  "roads": [
    {
      "id": "<R001, R002, ...>",
      "status": "<OPEN | CLOSED | RESTRICTED>",
      "location_desc": "<görüntüdeki bölge açıklaması, örn: sağ üst köşe>",
      "visual_evidence": "<gölge, moloz, araç, doku değişikliği vb.>",
      "confidence": "<HIGH | MEDIUM | LOW>",
      "recommended_action": "<eylem önerisi>"
    }
  ]
}"""

# Farklı senaryo için ek prompt'lar
BEFORE_AFTER_PROMPT = """Sana iki uydu görüntüsü sunuyorum: birincisi afet öncesi, ikincisi afet sonrası.

Her iki görüntüdeki yol ağını karşılaştırarak aşağıdaki analizi yap:

1. Afet sonrası KAPANAN yollar: Önceki görüntüde açık olan ama sonraki görüntüde engelli yollar.
2. Afet sonrası KISITLANAN yollar: Hasar gören veya daralan yollar.
3. AÇIK kalan yollar: Her iki görüntüde de erişilebilir yollar.

Görsel kanıtlar (moloz birikimi, zemin çökmesi, araç yığılması, su baskını) ile gerekçelendir.

Yanıtını YALNIZCA şu JSON şemasında ver:
{
  "comparison_summary": {
    "newly_closed": <int>,
    "newly_restricted": <int>,
    "remained_open": <int>,
    "damage_severity": "<LOW | MODERATE | HIGH | CRITICAL>"
  },
  "roads": [
    {
      "id": "<R001>",
      "pre_status": "<OPEN | CLOSED | RESTRICTED>",
      "post_status": "<OPEN | CLOSED | RESTRICTED>",
      "location_desc": "<bölge açıklaması>",
      "change_evidence": "<değişimi açıklayan görsel kanıt>",
      "confidence": "<HIGH | MEDIUM | LOW>"
    }
  ]
}"""

STATUS_COLOR = {
    "OPEN":       "#06d6a0",
    "CLOSED":     "#e63946",
    "RESTRICTED": "#f77f00",
}

STATUS_LABEL_TR = {
    "OPEN":       "Açık",
    "CLOSED":     "Kapalı",
    "RESTRICTED": "Kısıtlı",
}


# ── Dataclass'lar ────────────────────────────────────────────

@dataclass
class AIRoadStatus:
    """AI analizinden elde edilen tek yol kaydı."""
    road_id:            str
    status:             str        # OPEN / CLOSED / RESTRICTED
    location_desc:      str
    visual_evidence:    str
    confidence:         str        # HIGH / MEDIUM / LOW
    recommended_action: str = ""
    pre_status:         str = ""   # Before/After karşılaştırma için
    post_status:        str = ""

    @property
    def color(self) -> str:
        return STATUS_COLOR.get(self.status, "#adb5bd")

    @property
    def label_tr(self) -> str:
        return STATUS_LABEL_TR.get(self.status, self.status)

    def to_row(self) -> Dict:
        return {
            "ID":         self.road_id,
            "Durum":      self.label_tr,
            "Bölge":      self.location_desc,
            "Kanıt":      self.visual_evidence,
            "Güven":      self.confidence,
            "Öneri":      self.recommended_action,
        }


@dataclass
class AIAnalysisResult:
    """Claude Vision API tam analiz sonucu."""
    mode:            str           # "single" | "before_after"
    roads:           List[AIRoadStatus] = field(default_factory=list)
    summary:         Dict          = field(default_factory=dict)
    raw_response:    str           = ""
    model_used:      str           = ""
    input_tokens:    int           = 0
    output_tokens:   int           = 0

    @property
    def open_roads(self)       -> List[AIRoadStatus]:
        return [r for r in self.roads if r.status == "OPEN"]

    @property
    def closed_roads(self)     -> List[AIRoadStatus]:
        return [r for r in self.roads if r.status == "CLOSED"]

    @property
    def restricted_roads(self) -> List[AIRoadStatus]:
        return [r for r in self.roads if r.status == "RESTRICTED"]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "mode":         self.mode,
            "model":        self.model_used,
            "summary":      self.summary,
            "roads":        [r.to_row() for r in self.roads],
            "token_usage":  {"input": self.input_tokens, "output": self.output_tokens},
        }, ensure_ascii=False, indent=indent)


# ── Ana Sınıf ────────────────────────────────────────────────

class AIVisionAnalyzer:
    """
    Çok-backend AI Vision uydu görüntüsü yol analizi.

    Desteklenen backend'ler:
      "gemini"    → Google Gemini 2.0 Flash  (ÜCRETSİZ tier mevcut)
      "anthropic" → Claude Opus 4.6          (ücretli)

    Kullanım:
        # Gemini (ücretsiz):
        analyzer = AIVisionAnalyzer(api_key="AIza...", backend="gemini")

        # Claude (ücretli):
        analyzer = AIVisionAnalyzer(api_key="sk-ant-...", backend="anthropic")

        result = analyzer.analyze_single(image_array)
        result = analyzer.analyze_before_after(before_arr, after_arr)

    Gemini ücretsiz tier limitleri (2025):
        - 15 istek/dakika, 1 milyon token/dakika
        - gemini-2.0-flash-exp: görüntü analizi destekli
    """

    # Backend model adları
    GEMINI_MODEL    = "gemini-2.0-flash"
    ANTHROPIC_MODEL = "claude-opus-4-6"
    MAX_IMAGE_DIM   = 1568  # Her iki backend için max piksel

    def __init__(self, api_key: str, backend: str = "gemini"):
        if not api_key or not api_key.strip():
            raise ValueError("API anahtarı boş olamaz.")
        self._api_key = api_key.strip()
        self._backend = backend.lower()
        if self._backend not in ("gemini", "anthropic"):
            raise ValueError(f"Geçersiz backend: {backend}. 'gemini' veya 'anthropic' olmalı.")
        logger.debug(f"AIVisionAnalyzer başlatıldı — backend: {self._backend}")

    # ── Public API ──────────────────────────────────────────

    def analyze_single(
        self,
        image: np.ndarray,
        resolution_label: str = "0.5m/px (Planet SkySat)",
    ) -> AIAnalysisResult:
        """
        Tek görüntü yol durumu analizi.

        Args:
            image           : (H, W) veya (H, W, 3) numpy array
            resolution_label: Prompt'a eklenecek çözünürlük bilgisi

        Returns:
            AIAnalysisResult
        """
        logger.info("AI Vision: tek görüntü analizi başlatılıyor...")
        img_b64 = self._encode_image(image)

        prompt = ANALYSIS_PROMPT.replace(
            "0.5m çözünürlüklü", f"{resolution_label} çözünürlüklü"
        )

        response = self._call_api(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":       "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       img_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }]
        )

        return self._parse_single_response(response)

    def analyze_before_after(
        self,
        before: np.ndarray,
        after:  np.ndarray,
    ) -> AIAnalysisResult:
        """
        Before/After karşılaştırmalı yol analizi.

        Args:
            before: Afet öncesi görüntü array
            after : Afet sonrası görüntü array

        Returns:
            AIAnalysisResult (mode="before_after")
        """
        logger.info("AI Vision: before/after karşılaştırma analizi...")
        b64_before = self._encode_image(before)
        b64_after  = self._encode_image(after)

        response = self._call_api(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "**GÖRÜNTÜ 1 — Afet Öncesi:**"},
                    {
                        "type":   "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       b64_before,
                        },
                    },
                    {"type": "text", "text": "**GÖRÜNTÜ 2 — Afet Sonrası:**"},
                    {
                        "type":   "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       b64_after,
                        },
                    },
                    {"type": "text", "text": BEFORE_AFTER_PROMPT},
                ],
            }]
        )

        return self._parse_before_after_response(response)

    # ── İç Metodlar ─────────────────────────────────────────

    def _call_api(self, messages: List[Dict]) -> Dict:
        """Backend'e göre doğru API'yi çağırır."""
        if self._backend == "gemini":
            return self._call_gemini(messages)
        return self._call_anthropic(messages)

    def _call_gemini(self, messages: List[Dict]) -> Dict:
        """
        Google Gemini Vision API çağrısı.

        google-generativeai SDK: pip install google-generativeai
        Ücretsiz API key: https://aistudio.google.com/apikey
        """
        try:
            import google.generativeai as genai
            from PIL import Image as _PILImage
            import io as _io

            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(self.GEMINI_MODEL)

            # messages listesini Gemini content formatına çevir
            # Her content bloğu: text string veya PIL.Image
            content_parts = []
            for msg in messages:
                for block in msg.get("content", []):
                    if block.get("type") == "text":
                        content_parts.append(block["text"])
                    elif block.get("type") == "image":
                        src = block.get("source", {})
                        if src.get("type") == "base64":
                            import base64 as _b64
                            img_bytes = _b64.b64decode(src["data"])
                            pil_img = _PILImage.open(_io.BytesIO(img_bytes))
                            content_parts.append(pil_img)

            response = model.generate_content(content_parts)
            text_out = response.text if response.text else ""

            # Token sayısı (Gemini metadata'dan)
            usage = getattr(response, "usage_metadata", None)
            in_tok  = getattr(usage, "prompt_token_count", 0)    if usage else 0
            out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

            return {
                "content":       text_out,
                "model":         self.GEMINI_MODEL,
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
            }
        except ImportError:
            raise ImportError(
                "google-generativeai paketi gereklidir: "
                "pip install google-generativeai"
            )
        except Exception as exc:
            raise ConnectionError(f"Gemini API hatası: {exc}") from exc

    def _call_anthropic(self, messages: List[Dict]) -> Dict:
        """Anthropic Claude API çağrısı (ücretli)."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            msg = client.messages.create(
                model      = self.ANTHROPIC_MODEL,
                max_tokens = 2048,
                messages   = messages,
            )
            return {
                "content":       msg.content[0].text if msg.content else "",
                "model":         msg.model,
                "input_tokens":  msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            }
        except ImportError:
            raise ImportError("anthropic paketi gereklidir: pip install anthropic")
        except Exception as exc:
            raise ConnectionError(f"Anthropic API hatası: {exc}") from exc

    def _encode_image(self, image: np.ndarray) -> str:
        """numpy array → JPEG → base64 string."""
        from PIL import Image

        # Boyut sınırlama (Claude Vision max boyut)
        arr = image.copy()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]

        h, w = arr.shape[:2]
        max_d = self.MAX_IMAGE_DIM
        if max(h, w) > max_d:
            scale  = max_d / max(h, w)
            new_h  = int(h * scale)
            new_w  = int(w * scale)
            pil    = Image.fromarray(arr.astype(np.uint8)).resize(
                (new_w, new_h), Image.LANCZOS
            )
        else:
            pil = Image.fromarray(arr.astype(np.uint8))

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _parse_single_response(self, response: Dict) -> AIAnalysisResult:
        """API yanıtını AIAnalysisResult'a çevirir (tek görüntü modu)."""
        raw = response.get("content", "")
        roads: List[AIRoadStatus] = []
        summary: Dict = {}

        try:
            # JSON bloğunu çıkar
            data = self._extract_json(raw)
            summary = data.get("summary", {})
            for item in data.get("roads", []):
                roads.append(AIRoadStatus(
                    road_id           = item.get("id", f"R{len(roads)+1:03d}"),
                    status            = item.get("status", "OPEN").upper(),
                    location_desc     = item.get("location_desc", ""),
                    visual_evidence   = item.get("visual_evidence", ""),
                    confidence        = item.get("confidence", "MEDIUM").upper(),
                    recommended_action= item.get("recommended_action", ""),
                ))
        except Exception as exc:
            logger.warning(f"AI yanıt parse hatası: {exc} — ham metin döndürülüyor")

        logger.info(
            f"AI Vision (single): {len(roads)} yol — "
            f"Açık:{sum(1 for r in roads if r.status=='OPEN')} "
            f"Kapalı:{sum(1 for r in roads if r.status=='CLOSED')} "
            f"Kısıtlı:{sum(1 for r in roads if r.status=='RESTRICTED')}"
        )

        return AIAnalysisResult(
            mode          = "single",
            roads         = roads,
            summary       = summary,
            raw_response  = raw,
            model_used    = response.get("model", self.GEMINI_MODEL if self._backend == "gemini" else self.ANTHROPIC_MODEL),
            input_tokens  = response.get("input_tokens", 0),
            output_tokens = response.get("output_tokens", 0),
        )

    def _parse_before_after_response(self, response: Dict) -> AIAnalysisResult:
        """Before/After modu yanıt parse."""
        raw   = response.get("content", "")
        roads: List[AIRoadStatus] = []
        summary: Dict = {}

        try:
            data    = self._extract_json(raw)
            summary = data.get("comparison_summary", {})
            for item in data.get("roads", []):
                # Durum: post_status öncelikli
                status = item.get("post_status", item.get("status", "OPEN")).upper()
                roads.append(AIRoadStatus(
                    road_id           = item.get("id", f"R{len(roads)+1:03d}"),
                    status            = status,
                    location_desc     = item.get("location_desc", ""),
                    visual_evidence   = item.get("change_evidence",
                                                  item.get("visual_evidence", "")),
                    confidence        = item.get("confidence", "MEDIUM").upper(),
                    pre_status        = item.get("pre_status", "OPEN").upper(),
                    post_status       = status,
                ))
        except Exception as exc:
            logger.warning(f"AI before/after parse hatası: {exc}")

        return AIAnalysisResult(
            mode          = "before_after",
            roads         = roads,
            summary       = summary,
            raw_response  = raw,
            model_used    = response.get("model", self.GEMINI_MODEL if self._backend == "gemini" else self.ANTHROPIC_MODEL),
            input_tokens  = response.get("input_tokens", 0),
            output_tokens = response.get("output_tokens", 0),
        )

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """Metin içindeki ilk JSON bloğunu çıkarır."""
        import re
        # ```json ... ``` bloğunu ara
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # Doğrudan JSON
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError("JSON bulunamadı")
