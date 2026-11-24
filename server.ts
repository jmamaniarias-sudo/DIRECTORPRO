import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import os from 'os';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import JSZip from 'jszip';

const execFileAsync = promisify(execFile);

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;
const APP_BUILD_VERSION = '2026.11.24-v12';
const APP_BOOT_TIME = new Date().toISOString();

app.use(express.json({ limit: '10mb' }));

// Live app metadata endpoint (never cached) so ShareLinkModal always reflects the active server URL & build
app.get('/api/app-info', (_req: Request, res: Response) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  res.json({
    liveUrl: (process.env.APP_URL || '').trim() || null,
    buildVersion: APP_BUILD_VERSION,
    bootTime: APP_BOOT_TIME,
    mode: process.env.NODE_ENV || 'development',
  });
});

// Initialize GoogleGenAI client (only on server side)
// Use v1beta so Gemini 3.8 TTS speechConfig & speechMetadata are supported in @google/genai v3+
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = apiKey
  ? new GoogleGenAI({
      apiKey,
      httpOptions: {
        apiVersion: 'v1beta',
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

let cachedTextAiKey = '';
let cachedTextAi: GoogleGenAI | null = null;

function getTextAIClient(): GoogleGenAI | null {
  const currentKey = (process.env.GEMINI_API_KEY || '').trim();
  if (!currentKey) return null;
  if (cachedTextAi && cachedTextAiKey === currentKey) {
    return cachedTextAi;
  }
  cachedTextAiKey = currentKey;
  cachedTextAi = new GoogleGenAI({
    apiKey: currentKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
  return cachedTextAi;
}

// Chat endpoint with Gemini 4 Hydrogen
app.post('/api/ai/chat', async (req: Request, res: Response) => {
  try {
    const textAi = getTextAIClient();
    if (!textAi) {
      return res.status(503).json({
        error:
          'El servicio de Consultor con IA no tiene una clave GEMINI_API_KEY configurada en el servidor. Verifíquela en Settings > Secrets.',
      });
    }

    const { messages, documentContext } = req.body;
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Formato de mensajes inválido.' });
    }

    const systemInstruction = `Eres el "Consultor Pedagógico y Técnico DIRECTOR & CATEDRÁTICO PRO", un asistente de máxima especialización técnico-pedagógica para postulantes tanto al examen de institucionalización para el cargo de Director de Unidad Educativa como para la postulación al cargo de Docente Catedrático (Educación Superior / Formación de Maestros).

Posees conocimiento minucioso, exacto y contextual de las 26 fuentes oficiales fundamentales del examen:

--- BIBLIOGRAFÍA PARA DIRECTOR DE UNIDAD EDUCATIVA (GESTIÓN Y LIDERAZGO) ---
1. ODS 4 - UNESCO Educación 2030 (Declaración de Incheon y Marco de Acción):
   - 7 Metas de resultado (4.1 al 4.7): 12 años gratuitos (mínimo 9 obligatorios), preescolar obligatorio (4.2), equidad e inclusión (4.5), ciudadanía global (4.7).
   - 3 Medios de implementación (4.a al 4.c): entornos seguros (4.a), becas (4.b), docentes calificados (4.c). Financiamiento: 4%-6% PIB y 15%-20% gasto público.

2. LIDEI 2022 - OEI (Liderazgo Directivo Escolar en Iberoamérica):
   - Modelos: Liderazgo pedagógico/instruccional, Liderazgo distribuido, Liderazgo inclusivo y con justicia social, Liderazgo en contingencias.
   - Segundo factor intraescuela de mayor impacto en los aprendizajes. Superación de la carga burocrática hacia el acompañamiento al aula.

3. Marco para la Gestión y el Liderazgo Educativo Local (Ley 21.040, SLEP):
   - 6 Dimensiones directivas: Estratégico, Pedagógico, Personas/Profesional, Recursos, Convivencia Escolar y Vinculación Territorial.
   - Recursos Personales del Director: Principios éticos, habilidades de mediación/asertividad y conocimientos de PEL, PME y PEI.

4. Estudio Regional ERCE 2019 - UNESCO LLECE:
   - Diagnóstico en 16 países en 3° y 6° básica en Lectura, Matemática y Ciencias. Concentración en Nivel I.
   - Factores asociados: Clima escolar (+ impacto #1 intraescuela), educación preescolar (+), repitencia (- efecto devastador y abandono), ausentismo (-).

--- NORMATIVA ESPECÍFICA PARA DIRECTOR DE UNIDAD EDUCATIVA (BOLIVIA) ---
5. R.M. N° 0001/2026 (Normas Generales para la Gestión Educativa 2026 - Subsistema de Educación Regular, 79 Arts. + 11 DF):
   - Calendario 2026: 200 días hábiles en 3 trimestres; inscripciones y cierre de gestión. RUDE único y gratuito; prohibición de cobros indebidos, reservas de cupo o exámenes de ingreso.
   - Art. 29 II (Escala de Evaluación Inicial/Primaria/Secundaria): Ser (10 pts), Saber (45 pts), Hacer (40 pts), Decidir (5 pts) — suma 100 pts (distinguir de R.M. 0190/2024 Art. 49/56 que contempla Autoevaluación de 5 pts en Ser/Decidir).
   - Prohibición de expulsión de estudiantes embarazadas, sanciones sin debido proceso, tareas en feriados o discriminación. Obligatoriedad de Plan de Convivencia Pacífica y Armónica (PCPA).

6. R.M. N° 0864/2019 (09/08/2019 - Protocolo de Prevención, Actuación y Denuncia en Casos de Violencia Física, Psicológica y Sexual en UEs y CEE, abroga R.M. 2412/2017):
   - Plazos perentorios de actuación del Director: Denuncia obligatoria ante DNA/FELCV/Ministerio Público en un máximo de 24 horas desde conocido el hecho; informe a la Dirección Distrital dentro de las 48 horas; plan de contención y seguimiento en 72 horas.
   - Prohibición absoluta de conciliación, confrontación víctima-agresor, revictimización o traslado arbitrario de la víctima. Separación preventiva inmediata del agresor si es servidor público/docente conforme R.S. 212414 y D.S. 1302/1320.

7. Compilado de Apertura, Modificación y Cierre de Unidades Educativas Regular (R.M. 0020/2024 Fiscales RE-AMCUEF, R.M. 0021/2024 Convenio RE-FUEC, R.M. 0022/2024 Privadas RE-AMCUEP y R.M. 0534/2025 de 14/08/2025):
   - Registro de Unidades Educativas (RUE) y código SIE como requisito previo de funcionamiento legal. Trámite distrital -> DDE -> Ministerio de Educación.
   - R.M. 0534/2025 (modificatoria de UEs Privadas): infracciones leves, graves (multa del 10% del monto percibido por pensiones o sanción administrativa correspondiente) y gravísimas (clausura definitiva y cierre de RUE), prohibición de retención de libretas por mora y cobro de cuotas no autorizadas.

8. R.M. N° 0190/2024 (12/03/2024 - Reglamento de Evaluación del Desarrollo Curricular RE-EDC V.01, abroga R.M. 0473/2021; 74 Arts. + 7 DA + 1 DTU):
   - Evaluación integral, holística, continua, comunitaria y dialógica en 3 trimestres.
   - Art. 49 y Art. 56: Dimensiones Ser (5 pts), Saber (45 pts), Hacer (40 pts), Decidir (5 pts) + Autoevaluación del estudiante (5 pts en Ser/Decidir) = 100 pts; nota mínima de promoción anual: 51 puntos; reprobación por promedio menor a 51 o inasistencia injustificada.
   - Comisión Técnico Pedagógica (CTP) para detección temprana de dificultades y adaptaciones curriculares; prohibición de reprobación en Inicial y acompañamiento en Primaria según normativa vigente.

9. R.M. N° 0069/2025 (07/02/2025 - Reglamento para el Fortalecimiento de la Educación Inclusiva en el SEP - Modalidad Indirecta RPFEISEP-MI; 63 Arts. + 3 DF):
   - Modalidad Indirecta: inclusión de estudiantes con discapacidad, dificultades en el aprendizaje y talento extraordinario en Unidades Educativas Regulares mediante Adaptaciones Curriculares (de acceso, no significativas y significativas) y Equipos Multidisciplinarios de los CEE.
   - Art. 55 y Art. 56: Obligatoriedad de inscripción sin rechazo, emisión de informe psicopedagógico de apoyo del CEE y seguimiento del Director. Art. 17 inc. g): prohibición de discriminación o exigencia de acompañante pagado por la familia como requisito de matrícula.

10. R.M. N° 800/2015 (23/10/2015 - Reglamento de Emisión y Uso de Libretas y Certificados Electrónicos para Educación Alternativa, 34 Arts. + Manual SIEALT-MN-070-400):
   - Implementación del Sistema de Información Educativa de Educación Alternativa (SIEALT) y libretas/certificados electrónicos con código QR de seguridad y validación digital.
   - Competencias y responsabilidades del Director de CEA (verificación de registros, cierre de gestión en plataforma, emisión oportuna y gratuidad). Proscripción de alteración de notas en sistema bajo responsabilidad administrativa y penal.

11. D.S. N° 04688 (18/07/1957 - Reglamento del Escalafón Nacional del Servicio de Educación, XII Capítulos, 87 Artículos):
   - Enfoque Directivo y Docente: Funciones jerárquicas (Art. 14), calificación de méritos (Art. 20: a) estudios y títulos hasta 100 pts; b) condiciones personales y docentes hasta 100 pts; c) tiempo de servicios hasta 100 pts; d) méritos generales hasta 100 pts; rango f) Director de establecimiento), promociones de categoría (Cap. V, Arts. 29-39: Quinta 30%, Cuarta 45%, Tercera 60%, Segunda 75%, Primera 100%), e inamovilidad docente (Art. 54: ningún maestro inscrito puede ser destituido sin proceso previo).

12. R.S. N° 212414 (21/04/1993 - Reglamento de Faltas y Sanciones del Magisterio y Personal Docente y Administrativo, VIII Capítulos, 29 Artículos; abroga R.S. N° 203138 de 25/09/1990):
   - Derecho de defensa (Art. 3: invoca Art. 16 CPE, DUDH y Recomendación UNESCO de 05/10/1966). Art. 6 (Medidas precautorias): nadie puede ser destituido ni suspendido sin proceso previo, salvo el caso del Art. 11.a (reincidencia voluntaria en faltas GRAVES), complementado en normativa vigente por D.S. 1302/1320 (suspensión inmediata sin goce ante imputación formal por agresión/violencia sexual contra estudiantes).
   - Clasificación taxativa: Faltas Leves (Art. 9 - 8 incisos a-h; ej. 9.a suspensión de labores por cumpleaños y agasajos); Faltas Graves (Art. 10 - 22 incisos a-t incl. l, ll, n, ñ; ej. 10.d exacciones a padres, 10.e represalias, 10.g fiestas con bebidas alcohólicas en el establecimiento, 10.h abandono hasta 5d urbanos / 7d alejados sin licencia, 10.k apercibimiento humillante ante maestros/alumnos, 10.l entrega de documentos no públicos, 10.ll ineptitud administrativa manifiesta); Faltas Muy Graves (Art. 11 - 14 incisos a-m incl. l y ll; 11.a reincidencia voluntaria en faltas graves, 11.ll subalternización a intereses político-partidarios o sectarios, 11.m drogas, corrupción, acoso sexual, estupro, violencia, violación y bandas).
   - Sanciones (Art. 13 incisos a, b, c): a) Leves (amonestación en privado, amonestación escrita, descuento de 1 a 5 días, traslado); b) Graves (suspensión sin goce de 15 a 60 días, postergación de ascenso por 1 año, descenso a cargo inferior); c) Muy Graves (retiro definitivo del ejercicio del magisterio o destitución del cargo). Art. 14: prohibición de sanciones no previstas y remisión al Ministerio Público si hay indicios de delito.
   - Tribunales (Arts. 15-22): Art. 15 estructura tripartita (Presidente, Fiscal Promotor, Secretario-Actuario); Art. 16 Tribunal Nacional sede La Paz presidido por el Director General de Educación; Art. 17 Tribunales Departamentales/Regionales; Art. 18 maestros-abogados designados corresponsablemente con Federación/Confederación; Art. 19 igual o superior jerarquía a los encauzados; Art. 20 reemplazo inmediato; Art. 21 excusas y recusaciones SOLO por parentesco consanguíneo hasta 4° grado o vínculo espiritual determinado. Actualización vigente (D.S. 25273 y D.S. 23968): 1.ª instancia Tribunal Distrital y 2.ª instancia Tribunal Departamental unipersonal (Director Departamental de Educación).
   - Procedimiento y Archivo (Arts. 23-29): Art. 23 denuncia verbal (sienta acta) o escrita (firma o impresión digital con presencia de un testigo) ante superior inmediato; Art. 24 los 8 términos (a: 48h remisión; b: 24h citación; c: 20d probatorio prorrogable por distancia; d: 24h notificaciones; e: 5d fallo; f: 3d apelación; g: 48h remisión obrados; h: 24h toda otra actuación); Art. 25 apelación en 3 días con agravios ante el MISMO tribunal que sentenció la causa; Art. 26 fallo de apelación en 15 días; Art. 27 emisión por Ministerio de Educación mediante Resolución Ministerial con autoridad de cosa juzgada; Art. 28 ejecución en 3 días; Art. 29 custodia en Departamento de Archivo del MinEdu más copia legalizada al expediente personal y a autoridades superiores inmediatas.

--- BIBLIOGRAFÍA PARA DOCENTE CATEDRÁTICO ---
5. Neuroeducación: solo se puede aprender aquello que se ama (Francisco Mora):
   - Axioma central: La emoción es el vehículo biológico imprescindible del aprendizaje; activa la amígdala e hipocampo encendiendo la corteza cerebral.
   - Curiosidad y dopamina: La novedad activa el área tegmental ventral proyectando dopamina al núcleo accumbens y la corteza prefrontal.
   - Límites atencionales: La atención ejecutiva sostenida decae a los 15-20 minutos; exige rupturas atencionales intencionadas y silencio pedagógico.
   - Consolidación en el sueño: En el sueño No-REM de ondas lentas y REM se transfiere la memoria del hipocampo a la neocorteza y ocurre la poda sináptica.
   - Refutación de Neuromitos: Desmiente los estilos de aprendizaje VAK, el mito del 10% del cerebro y la falsa división de hemisferios. El cerebro es multisensorial y plástico.
   - Funciones ejecutivas: Memoria de trabajo, control inhibitorio y flexibilidad en corteza prefrontal (maduración hasta los 20-25 años).

6. Didáctica General y Formación del Profesorado (Ernesto López Gómez coord. - UNIR):
   - Estatuto epistemológico: Ciencia teórico-práctica que optimiza los procesos de enseñanza-aprendizaje. Transposición didáctica (Chevallard).
   - 4 Niveles de concreción curricular: 1° Diseño Curricular Base nacional, 2° Proyecto de Centro (PEI/PEC), 3° Programación de Aula / Guía Docente, 4° Adaptaciones (ACI).
   - 7 Principios didácticos: Individualización, socialización, actividad (aprender haciendo), intuición, juego/creatividad, globalización y apertura.
   - Métodos activos: Kilpatrick (proyectos de producción, consumo, problema y perfeccionamiento), ABP (PBL) y Estudio de Casos.
   - Aprendizaje cooperativo (Johnson & Johnson): 5 pilares obligatorios (interdependencia positiva, interacción cara a cara, responsabilidad individual, habilidades sociales, procesamiento grupal). Técnicas Jigsaw (Aronson) y STAD (Slavin). Modelo RICA de TIC.
   - Evaluación auténtica: Diagnóstica, formativa (procesual reguladora con retroalimentación) y formadora con rúbricas analíticas.
   - Profesor reflexivo: Donald Schön (conocimiento, reflexión en y sobre la acción) y espiral de investigación-acción de Elliott (planificar, actuar, observar, reflexionar).

7. Metodología de la Investigación Cualitativa Interpretativa (Arturo Barraza Macías):
   - Paradigma interpretativo: Hermenéutica (círculo hermenéutico), Fenomenología (Lebenswelt), Interaccionismo Simbólico y Etnometodología. Comprensión (Verstehen).
   - Perspectiva Émica (emic: óptica interna de los actores) vs Ética (etic: categorías externas). Indexicabilidad y reflexividad del investigador.
   - Técnicas: Observación participante y Diario de Campo (notas descriptivas factuales, notas metodológicas y notas teóricas/reflexivas), Entrevista en profundidad y Grupos de Discusión.
   - Reducción inductiva de datos: Transcripción verbatim -> Codificación abierta -> Codificación axial/categorización -> Modelo de relaciones intercategoriales hasta la Saturación Teórica.
   - 4 Criterios de Rigor Científico (Lincoln y Guba): Credibilidad (validez interna cualitativa por triangulación y member checking), Transferibilidad (descripción densa contextual), Dependencia (consistencia del proceso) y Confirmabilidad (pista de auditoría documental que vincula hallazgos a testimonios primarios).

--- NORMATIVA ESPECÍFICA PARA DOCENTE CATEDRÁTICO (BOLIVIA) ---
8. D.S. N° 04688 (18 de julio de 1957) - Reglamento del Escalafón Nacional del Servicio de Educación (XII Capítulos, 87 Artículos):
   - Registro sistemático, permanente y centralizado de datos personales y profesionales (Cap. II). Requisitos de ingreso docente (Art. 7): 1) Ser boliviano o extranjero contratado; 2) Acreditar capacidad para el desempeño (normalista/titular); 3) Tener no menos de 18 ni más de 50 años.
   - Clasificación del personal (Cap. III): Docente (normalistas, titulares por antigüedad con 10 años, interinos) y Administrativo. Jerarquía docente (Art. 14).
   - Calificación de Méritos (Art. 20) sobre 400 puntos máximos: a) Estudios y títulos (hasta 100 pts); b) Condiciones personales y docentes (hasta 100 pts); c) Tiempo de servicios (hasta 100 pts); d) Méritos generales (hasta 100 pts).
   - Promociones de categoría (Cap. V, Arts. 29-39): Quinta (30%), Cuarta (45%), Tercera (60%), Segunda (75%) y Primera (100%).
   - Inamovilidad docente (Art. 54) y causales de pérdida de derechos (Art. 73).

9. Compendio de Normativa para las ESFM y Unidades Académicas (Versión 2018 - Ministerio de Educación, R.M. 2726/2017 y R.M. 2938/2017, 520 pp.):
   - 11 Reglamentos Oficiales + Manual de Organización y Funciones (MOF, 26 cargos; Cargo 7: Docente de Especialidad/Formación General, dependencia del Director Académico, Art. 58 Ley 070).
   - Reglamento de Evaluación del Proceso Formativo: Asistencia obligatoria mínima de 85% (100% en PEC). Escala de 1 a 100 puntos; nota mínima de aprobación: 60 puntos. Segunda instancia: máximo en 2 Unidades de Formación (UF) con promedio entre 45 y 59 puntos, dentro de los 3 días hábiles posteriores; nota máxima en segunda instancia: 60 puntos.
   - Reglamento de la Práctica Educativa Comunitaria (PEC): 540 horas académicas distribuidas progresivamente en los 5 años de formación.
   - Reglamento de Modalidades de Graduación (5 años / Licenciatura): Excelencia Académica (>= 90 pts sin reprobaciones + artículo académico), Proyecto Sociocomunitario Productivo, Sistematización de Experiencias Educativas, y Proyecto de Grado. Defensa pública de 50 minutos ante CCETG.
   - Reglamento de Licencias, Permisos y Abandono (6 días continuos o 10 discontinuos en el mes), CESC, Becas, Bajas y Traspasos, Idioma Originario/LSB, Diplomas/Títulos (17 requisitos, 10 días hábiles), y TACFI (48 horas).

--- NORMATIVA GENERAL (COMÚN PARA AMBOS CARGOS - BOLIVIA) ---
11. Constitución Política del Estado Plurinacional de Bolivia (CPE, 2009):
   - Estado Unitario Social de Derecho Plurinacional Comunitario (Art. 1). 36 idiomas indígenas + castellano (Art. 5). Principios ético-morales: ama qhilla, ama llulla, ama suwa, suma qamaña (Art. 8).
   - Educación como función suprema y primera responsabilidad financiera del Estado (Art. 77); unitaria, pública, universal, democrática, participativa, comunitaria, descolonizadora y de calidad (Art. 78); obligatoria hasta el bachillerato y gratuita hasta el nivel superior (Art. 81). ESFM únicas fiscales y gratuitas (Art. 96).
   - Acciones de Defensa (Arts. 125-136). Función Pública (Arts. 232-239) y primacía constitucional (Art. 410).

12. Ley N° 070 de la Educación "Avelino Siñani - Elizardo Pérez" (2010):
   - Bases, fines y objetivos del SEP (Arts. 1-5). Intraculturalidad, interculturalidad, plurilingüismo (Art. 7). 3 Subsistemas (Regular, Alternativa y Especial, Superior). MESCP y participación social comunitaria.

13. Ley N° 045 Contra el Racismo y Toda Forma de Discriminación (2010):
   - Principios (Art. 2), definiciones (Art. 5), reglamentos internos en educación y delitos del Código Penal (Arts. 281 quinquies a nonies).

14. Ley N° 348 Integral para Garantizar a las Mujeres una Vida Libre de Violencia (2013):
   - 16 tipos de violencia (Art. 7), medidas en educación (Art. 19), prohibición de conciliación (Art. 46), denuncia inmediata ante FELCV/SLIM/Ministerio Público.

15. Ley N° 548 Código Niña, Niño y Adolescente (2014):
   - Interés superior y prioridad absoluta; violencia en el sistema educativo (Arts. 150-152); prohibición de castigo físico/humillante; PCPA; denuncia ante DNA en 24 horas; prohibición de expulsión de estudiantes embarazadas.

16. Ley N° 223 Ley General para Personas con Discapacidad (2012):
   - Educación inclusiva con adaptaciones curriculares, accesibilidad, LSB y Braille; cuota laboral (4% público / 2% privado) e inamovilidad laboral (Art. 34).

17. Ley N° 1178 de Administración y Control Gubernamentales (SAFCO, 1990):
   - 8 Sistemas (Programar/organizar: SPO, SOA, SP; Ejecutar: SAP, SABS, STyCP, SCI; Controlar: Control Gubernamental).
   - 4 Tipos de Responsabilidad (Arts. 28-38): Administrativa (2 años; multa hasta 20%, suspensión hasta 30d o destitución), Ejecutiva (2 años), Civil (imprescriptible según Art. 324 CPE) y Penal.

18. Ley N° 004 de Lucha Contra la Corrupción "Marcelo Quiroga Santa Cruz" (2010):
   - Imprescriptibilidad de delitos de corrupción con grave daño económico (Art. 112 CPE), inexistencia de fuero, deber de denuncia y tipos penales de corrupción.

Instrucciones de Respuesta:
- Responde siempre en español formal, riguroso y pedagógico.
- Identifica si la consulta alude a Dirección Escolar, Docente Catedrático o Normativa General, y fundamenta con precisión sin confundir artículos, leyes, autores ni capítulos entre sí.
- Si el usuario te consulta sobre una pregunta de examen o dilema institucional/legal, analiza las alternativas, explica cuál es la correcta según la norma o literatura científica y por qué las distractoras son erróneas.
- Mantén un tono académico de alto nivel, claro, bien estructurado y enfocado en que el postulante obtenga la máxima nota.`;

    // Normalize conversation turns: filter empty, start with 'user', merge adjacent same-role turns
    const rawTurns = messages
      .filter(
        (m: any) =>
          m &&
          typeof m.content === 'string' &&
          m.content.trim().length > 0 &&
          (m.role === 'user' || m.role === 'assistant')
      )
      .map((m: { role: string; content: string }) => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        text: m.content.trim(),
      }));

    // Drop any leading model greeting turns so contents always begins with 'user'
    const firstUserIdx = rawTurns.findIndex((t) => t.role === 'user');
    if (firstUserIdx === -1) {
      return res.status(400).json({ error: 'Debe incluir al menos una consulta del usuario.' });
    }

    const normalizedTurns: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];
    for (const turn of rawTurns.slice(firstUserIdx)) {
      const prev = normalizedTurns[normalizedTurns.length - 1];
      if (prev && prev.role === turn.role) {
        prev.parts.push({ text: turn.text });
      } else {
        normalizedTurns.push({
          role: turn.role as 'user' | 'model',
          parts: [{ text: turn.text }],
        });
      }
    }

    if (documentContext && typeof documentContext === 'string' && documentContext.trim()) {
      const lastTurn = normalizedTurns[normalizedTurns.length - 1];
      if (lastTurn && lastTurn.role === 'user') {
        lastTurn.parts.unshift({
          text: `[Contexto del documento en consulta: ${documentContext.trim()}]\n\n`,
        });
      }
    }

    const candidateModels = ['gemini-4-hydrogen', 'gemini-3.1-flash-lite'] as const;
    let generatedText = '';
    let lastError: any = null;

    for (const modelName of candidateModels) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const response = await textAi.models.generateContent({
            model: modelName,
            contents: normalizedTurns,
            config: {
              systemInstruction,
              ...(modelName === 'gemini-4-hydrogen'
                ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
                : {}),
            },
          });
          if (response.text && response.text.trim()) {
            generatedText = response.text.trim();
            break;
          }
        } catch (err: any) {
          lastError = err;
          const status = err?.status ?? err?.code;
          // Do not retry on invalid API key / permission denied
          if (status === 400 || status === 401 || status === 403) {
            break;
          }
          if (attempt === 0) {
            await new Promise((r) => setTimeout(r, 600));
          }
        }
      }
      if (generatedText) break;
      const status = lastError?.status ?? lastError?.code;
      if (status === 401 || status === 403) break;
    }

    if (!generatedText) {
      const status = lastError?.status ?? lastError?.code;
      if (status === 401 || status === 403) {
        return res.status(403).json({
          error:
            'La clave de API de Gemini fue rechazada. Por favor verifique la clave en Settings > Secrets.',
        });
      }
      if (status === 429) {
        return res.status(429).json({
          error:
            'Se alcanzó temporalmente el límite de consultas del servicio de IA. Intente nuevamente en unos segundos.',
        });
      }
      throw lastError || new Error('No se recibió respuesta del modelo de IA.');
    }

    res.json({ text: generatedText });
  } catch (error: any) {
    console.error('Error en /api/ai/chat:', error);
    res.status(500).json({
      error: error?.message || 'Error procesando la consulta con el Consultor IA.',
    });
  }
});

// =========================================================================
// STRICT 43-PDF NORMATIVE SCRIPT STORE & PERSISTENT GEMINI TTS WAV CACHE
// =========================================================================
const DATA_DIR = path.join(__dirname, 'data');
const NORMATIVE_SCRIPTS_FILE = path.join(DATA_DIR, 'normative-verified-scripts.json');
const NORMATIVE_TTS_DIR = path.join(DATA_DIR, 'normative-tts-cache');
const NORMATIVE_TTS_INDEX_FILE = path.join(DATA_DIR, 'normative-tts-index.json');

interface PersistedSlideScript {
  page: number;
  title: string;
  keyTakeaway: string;
  narration: string;
  verifiedAt: string;
  source: 'pdf-batch' | 'canvas-visual';
}

interface PersistedAudioEntry {
  filename: string;
  textHash: string;
  voiceName: string;
  generatedAt: string;
  bytes: number;
}

type PersistedScriptsStore = Record<string, Record<string, PersistedSlideScript>>;
type PersistedAudioIndex = Record<string, PersistedAudioEntry>;

const ALLOWED_TTS_VOICES = new Set(['Kore', 'Puck', 'Charon', 'Fenrir', 'Aoede']);

function sanitizeVoiceName(voice?: string): string {
  return voice && ALLOWED_TTS_VOICES.has(voice) ? voice : 'Kore';
}

function sanitizeIdPart(raw: string): string {
  return raw.replace(/[^a-zA-Z0-9_-]/g, '');
}

function computeTextHash(text: string, voiceName: string): string {
  return crypto
    .createHash('sha256')
    .update(`${voiceName}::${text.trim()}`)
    .digest('hex')
    .slice(0, 16);
}

function ensureDataDirs(): void {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  if (!fs.existsSync(NORMATIVE_TTS_DIR)) {
    fs.mkdirSync(NORMATIVE_TTS_DIR, { recursive: true });
  }
}

function loadScriptsStore(): PersistedScriptsStore {
  try {
    if (fs.existsSync(NORMATIVE_SCRIPTS_FILE)) {
      const raw = fs.readFileSync(NORMATIVE_SCRIPTS_FILE, 'utf-8');
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        return parsed as PersistedScriptsStore;
      }
    }
  } catch (err) {
    console.error('Error leyendo normative-verified-scripts.json:', err);
  }
  return {};
}

function saveScriptsStore(store: PersistedScriptsStore): void {
  ensureDataDirs();
  const tmpFile = `${NORMATIVE_SCRIPTS_FILE}.tmp`;
  fs.writeFileSync(tmpFile, JSON.stringify(store, null, 2), 'utf-8');
  fs.renameSync(tmpFile, NORMATIVE_SCRIPTS_FILE);
}

function loadAudioIndex(): PersistedAudioIndex {
  try {
    if (fs.existsSync(NORMATIVE_TTS_INDEX_FILE)) {
      const raw = fs.readFileSync(NORMATIVE_TTS_INDEX_FILE, 'utf-8');
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        return parsed as PersistedAudioIndex;
      }
    }
  } catch (err) {
    console.error('Error leyendo normative-tts-index.json:', err);
  }
  return {};
}

function saveAudioIndex(index: PersistedAudioIndex): void {
  ensureDataDirs();
  const tmpFile = `${NORMATIVE_TTS_INDEX_FILE}.tmp`;
  fs.writeFileSync(tmpFile, JSON.stringify(index, null, 2), 'utf-8');
  fs.renameSync(tmpFile, NORMATIVE_TTS_INDEX_FILE);
}

let normativeScriptsStore: PersistedScriptsStore = loadScriptsStore();
let normativeAudioIndex: PersistedAudioIndex = loadAudioIndex();

// In-memory LRU TTS audio cache for non-persisted clips (capped to avoid unbounded RAM)
const MAX_MEM_TTS_CLIPS = 50;
const ttsAudioCache = new Map<string, Buffer>();

function setMemTtsCache(key: string, buf: Buffer): void {
  if (ttsAudioCache.has(key)) {
    ttsAudioCache.delete(key);
  } else if (ttsAudioCache.size >= MAX_MEM_TTS_CLIPS) {
    const oldestKey = ttsAudioCache.keys().next().value;
    if (oldestKey) ttsAudioCache.delete(oldestKey);
  }
  ttsAudioCache.set(key, buf);
}

function isRateLimitError(err: any): boolean {
  const msg = String(err?.message || err || '');
  const status = Number(err?.status || err?.code || 0);
  return status === 429 || msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED');
}

function parseRetryDelayMs(err: any): number {
  const msg = String(err?.message || err || '');
  const matchSec =
    /retryDelay["']?\s*:\s*["']?(\d+(?:\.\d+)?)s/i.exec(msg) ||
    /Please retry in\s+(\d+(?:\.\d+)?)s/i.exec(msg);
  if (matchSec) {
    const parsedMs = Math.ceil(parseFloat(matchSec[1]) * 1000) + 400;
    return Math.min(Math.max(parsedMs, 1500), 15000);
  }
  return 2500;
}

function isTransientGeminiError(err: any): boolean {
  const msg = String(err?.message || err || '');
  const status = Number(err?.status || err?.code || 0);
  return (
    status === 503 ||
    status === 429 ||
    status === 500 ||
    msg.includes('503') ||
    msg.includes('UNAVAILABLE') ||
    msg.includes('429') ||
    msg.includes('RESOURCE_EXHAUSTED') ||
    msg.includes('high demand')
  );
}

async function sleepMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runWithModelFallback<T>(
  models: string[],
  executor: (model: string) => Promise<T>
): Promise<T> {
  let lastError: any = null;
  for (const model of models) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        return await executor(model);
      } catch (err: any) {
        lastError = err;
        if (!isTransientGeminiError(err)) {
          throw err;
        }
        // On 429 with long retryDelay (>10s), immediately move to next model instead of stalling
        if (isRateLimitError(err)) {
          const delayMs = parseRetryDelayMs(err);
          if (delayMs > 10000 && model !== models[models.length - 1]) {
            break;
          }
          await sleepMs(delayMs);
        } else {
          await sleepMs(attempt * 1200);
        }
      }
    }
  }
  throw lastError || new Error('El servicio de Gemini está temporalmente ocupado.');
}

function sendGeminiRouteError(
  res: Response,
  routeLabel: string,
  error: any,
  fallbackMessage: string
) {
  if (isRateLimitError(error)) {
    const retrySec = Math.ceil(parseRetryDelayMs(error) / 1000);
    console.warn(`[RateLimit 429] ${routeLabel} — reintentar en ~${retrySec}s`);
    return res.status(429).json({
      error: `Límite temporal de solicitudes por minuto alcanzado. Espera aproximadamente ${retrySec} segundos e intenta nuevamente.`,
      retryAfterSeconds: retrySec,
    });
  }
  console.error(`Error en ${routeLabel}:`, error?.message || error);
  return res.status(500).json({
    error: error?.message || fallbackMessage,
  });
}

async function synthesizeGeminiWav(text: string, voiceName = 'Kore'): Promise<Buffer> {
  if (!ai) {
    throw new Error('API Key de Gemini no configurada en el servidor.');
  }
  const safeVoice = sanitizeVoiceName(voiceName);
  const cleanedText = text
    .replace(/[*#_`>]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 3800);

  if (!cleanedText) {
    throw new Error('Texto vacío para síntesis vocal.');
  }

  const response = await runWithModelFallback(
    ['gemini-3.8-flash-lite-tts', 'gemini-3.8-flash-tts'],
    (model) =>
      ai.models.generateContent({
        model,
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: cleanedText,
                speechMetadata: {
                  style:
                    'Relator académico claro, pausado, profesional y pedagógico en español latinoamericano',
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: safeVoice,
              },
            },
          },
        },
      })
  );

  const parts = response.candidates?.[0]?.content?.parts ?? [];
  const audioPart = parts.find((p: any) => p?.inlineData?.data);
  if (!audioPart?.inlineData?.data) {
    throw new Error('Gemini TTS no devolvió datos de audio.');
  }

  return Buffer.from(audioPart.inlineData.data, 'base64');
}

function persistNormativeSlideAudio(
  slideKey: string,
  text: string,
  voiceName: string,
  wavBuffer: Buffer
): PersistedAudioEntry {
  ensureDataDirs();
  const safeKey = sanitizeIdPart(slideKey);
  const safeVoice = sanitizeVoiceName(voiceName);
  const textHash = computeTextHash(text, safeVoice);
  const filename = `${safeKey}.wav`;
  const fullPath = path.join(NORMATIVE_TTS_DIR, filename);
  fs.writeFileSync(fullPath, wavBuffer);

  const entry: PersistedAudioEntry = {
    filename,
    textHash,
    voiceName: safeVoice,
    generatedAt: new Date().toISOString(),
    bytes: wavBuffer.length,
  };
  normativeAudioIndex[safeKey] = entry;
  saveAudioIndex(normativeAudioIndex);
  return entry;
}

function getPersistedSlideAudioIfFresh(
  slideKey: string,
  text: string,
  voiceName?: string
): Buffer | null {
  const safeKey = sanitizeIdPart(slideKey);
  const entry = normativeAudioIndex[safeKey];
  if (!entry) return null;
  const targetVoice = voiceName ? sanitizeVoiceName(voiceName) : entry.voiceName;
  const expectedHash = computeTextHash(text, targetVoice);
  if (entry.textHash !== expectedHash) return null;
  const fullPath = path.join(NORMATIVE_TTS_DIR, entry.filename);
  if (!fs.existsSync(fullPath)) return null;
  return fs.readFileSync(fullPath);
}

function resolveNormativePdfFile(pdfPathRel: string): string | null {
  if (!pdfPathRel || typeof pdfPathRel !== 'string') return null;
  const prefix = '/assets/aistudio/';
  if (!pdfPathRel.startsWith(prefix)) return null;
  const relName = decodeURIComponent(pdfPathRel.slice(prefix.length)).replace(/^\/+/, '');
  if (!relName || relName.includes('..') || relName.includes('\0') || !relName.toLowerCase().endsWith('.pdf')) {
    return null;
  }

  const candidateDirs = [
    path.join(__dirname, 'public', 'assets', 'aistudio'),
    path.join(__dirname, 'dist', 'assets', 'aistudio'),
  ];

  for (const baseDir of candidateDirs) {
    const normalizedBase = path.resolve(baseDir);
    const candidatePath = path.resolve(normalizedBase, relName);
    if (
      candidatePath.startsWith(normalizedBase + path.sep) &&
      fs.existsSync(candidatePath) &&
      fs.statSync(candidatePath).isFile()
    ) {
      return candidatePath;
    }
  }
  return null;
}

async function renderPdfPagesToJpeg(
  absPdfPath: string,
  startPage: number,
  endPage: number
): Promise<Array<{ page: number; base64: string }>> {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'norm-slide-'));
  try {
    const outPattern = path.join(tmpDir, 'slide-%03d.jpg');
    await execFileAsync(
      'gs',
      [
        '-q',
        '-dNOPAUSE',
        '-dBATCH',
        '-dSAFER',
        '-sDEVICE=jpeg',
        '-dJPEGQ=80',
        '-r100',
        `-dFirstPage=${startPage}`,
        `-dLastPage=${endPage}`,
        `-sOutputFile=${outPattern}`,
        absPdfPath,
      ],
      { timeout: 60000, maxBuffer: 16 * 1024 * 1024 }
    );

    const files = fs
      .readdirSync(tmpDir)
      .filter((f) => f.endsWith('.jpg'))
      .sort();

    return files.map((file, idx) => {
      const buf = fs.readFileSync(path.join(tmpDir, file));
      return {
        page: startPage + idx,
        base64: buf.toString('base64'),
      };
    });
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

async function extractStrictSlideBatch(
  deckTitle: string,
  totalSlides: number,
  slidesBatch: Array<{ page: number; base64: string }>
): Promise<Array<{ page: number; title: string; keyTakeaway: string; narration: string }>> {
  if (!ai) {
    throw new Error('API Key de Gemini no configurada en el servidor.');
  }

  const pageNumbersList = slidesBatch.map((s) => s.page).join(', ');
  const parts: any[] = [];

  for (const slide of slidesBatch) {
    parts.push({
      text: `[IMAGEN DE LA DIAPOSITIVA ${slide.page} DE ${totalSlides}]:`,
    });
    parts.push({
      inlineData: {
        mimeType: 'image/jpeg',
        data: slide.base64,
      },
    });
  }

  parts.push({
    text: `Eres un expositor académico experto en legislación educativa boliviana que presenta estas diapositivas en vivo ante postulantes a cargos directivos y docentes.
Analiza EXCLUSIVAMENTE las imágenes adjuntas correspondientes a las diapositivas ${pageNumbersList} de ${totalSlides} de la presentación "${deckTitle}".

REGLAS ESTRICTAS DE FIDELIDAD Y ESTILO EXPOSITOR:
1. EXPOSICIÓN FIEL SIN INVENTAR: Explica ÚNICAMENTE lo que aparece visible en cada lámina (título exacto, subtítulos, cuadros, diagramas, columnas, flechas, fechas, artículos, cifras, citas y viñetas reales de esa diapositiva). PROHIBIDO añadir artículos, leyes o conceptos externos que no estén en la imagen.
2. ESTILO DE EXPOSITOR EN VIVO: Habla con tono claro, fluido, pedagógico y natural, como un expositor explicando la diapositiva proyectada frente a un auditorio ("En esta lámina observamos...", "Como muestra el esquema...", "El punto central que destaca esta diapositiva es..."). Conecta con naturalidad cada sección o bloque visual de la lámina.
3. TRADUCCIÓN FIEL: Si alguna diapositiva contiene texto en inglés, tradúcelo y explícalo con total naturalidad al español conservando exactamente los mismos datos, títulos y estructura visual.
4. TEXTO LIMPIO PARA VOZ (TTS): El campo "narration" debe estar redactado en prosa continua lista para ser leída en voz alta al instante, SIN asteriscos, SIN guiones de lista, SIN barras ni símbolos Markdown. Escribe abreviaturas legales de forma natural cuando ayude a la lectura ("Artículo", "Ley número", "Resolución Ministerial").
5. APERTURA EXACTA: Cada "narration" DEBE comenzar con "Diapositiva X de ${totalSlides}: [Título exacto]. " seguido de la explicación expositiva completa (entre 70 y 150 palabras por lámina según la densidad visual de la diapositiva, sin omitir datos importantes).
6. Devuelve exactamente un elemento en "slides" por cada página solicitada (${pageNumbersList}), respetando su número exacto de página.`,
  });

  const response = await runWithModelFallback(
    ['gemini-4-hydrogen', 'gemini-3.1-flash-lite'],
    (model) =>
      ai.models.generateContent({
        model,
        contents: [{ role: 'user', parts }],
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              slides: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    page: { type: Type.INTEGER },
                    title: { type: Type.STRING },
                    keyTakeaway: { type: Type.STRING },
                    narration: { type: Type.STRING },
                  },
                  required: ['page', 'title', 'keyTakeaway', 'narration'],
                },
              },
            },
            required: ['slides'],
          },
        },
      })
  );

  const rawText = (response.text || '').trim();
  const parsed = JSON.parse(rawText) as {
    slides?: Array<{ page: number; title: string; keyTakeaway: string; narration: string }>;
  };
  const extracted = Array.isArray(parsed.slides) ? parsed.slides : [];

  return slidesBatch.map((s, idx) => {
    const match = extracted.find((e) => Number(e.page) === s.page) ?? extracted[idx];
    const cleanTitle = (match?.title || `Diapositiva ${s.page}`).trim();
    const cleanTakeaway = (
      match?.keyTakeaway ||
      `Contenido extraído de la diapositiva ${s.page} de ${totalSlides}.`
    ).trim();
    let cleanNarration = (match?.narration || '')
      .replace(/[*#_`]/g, '')
      .replace(/\s+/g, ' ')
      .trim();

    if (!cleanNarration) {
      cleanNarration = `Diapositiva ${s.page} de ${totalSlides}: ${cleanTitle}. ${cleanTakeaway}`;
    } else if (!/^diapositiva\s+\d+/i.test(cleanNarration)) {
      cleanNarration = `Diapositiva ${s.page} de ${totalSlides}: ${cleanTitle}. ${cleanNarration}`;
    }

    return {
      page: s.page,
      title: cleanTitle,
      keyTakeaway: cleanTakeaway,
      narration: cleanNarration,
    };
  });
}

// Text-to-Speech endpoint with Gemini 3.8 Flash Lite TTS
app.post('/api/ai/tts', async (req: Request, res: Response) => {
  try {
    if (!ai) {
      return res.status(500).json({
        error: 'API Key de Gemini no configurada.',
      });
    }

    const { text, blockId, voiceName } = req.body;
    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'Texto requerido para síntesis vocal.' });
    }

    const safeVoice = sanitizeVoiceName(voiceName);
    const cleanedText = text.slice(0, 3800).trim();
    const isPersistedBlock =
      typeof blockId === 'string' &&
      (blockId.startsWith('norm-') || blockId.startsWith('biblio_'));

    // 1. Check persistent disk cache first for normative slides & bibliography book blocks
    if (isPersistedBlock) {
      const diskBuf = getPersistedSlideAudioIfFresh(blockId, cleanedText, voiceName);
      if (diskBuf) {
        res.set({
          'Content-Type': 'audio/wav',
          'Content-Length': diskBuf.length,
          'Cache-Control': 'public, max-age=86400',
        });
        return res.send(diskBuf);
      }
    }

    // 2. Check in-memory LRU cache keyed by blockId + textHash
    const textHash = computeTextHash(cleanedText, safeVoice);
    const cacheKey = blockId ? `${blockId}__${textHash}` : textHash;
    if (ttsAudioCache.has(cacheKey)) {
      const cached = ttsAudioCache.get(cacheKey)!;
      res.set({
        'Content-Type': 'audio/wav',
        'Content-Length': cached.length,
        'Cache-Control': 'public, max-age=86400',
      });
      return res.send(cached);
    }

    // 3. Synthesize fresh WAV audio via Gemini 3.8 Flash Lite TTS
    const audioBuffer = await synthesizeGeminiWav(cleanedText, safeVoice);

    if (isPersistedBlock) {
      persistNormativeSlideAudio(blockId, cleanedText, safeVoice, audioBuffer);
    } else {
      setMemTtsCache(cacheKey, audioBuffer);
    }

    res.set({
      'Content-Type': 'audio/wav',
      'Content-Length': audioBuffer.length,
      'Cache-Control': 'public, max-age=86400',
    });
    return res.send(audioBuffer);
  } catch (error: any) {
    return sendGeminiRouteError(res, '/api/ai/tts', error, 'Error en síntesis vocal.');
  }
});

// Status endpoint: returns all persisted verified scripts & cached TTS WAV metadata
app.get('/api/normative/status', (_req: Request, res: Response) => {
  res.json({
    aiConfigured: Boolean(ai),
    scripts: normativeScriptsStore,
    audioIndex: normativeAudioIndex,
  });
});

// Process a batch of 1 to 4 slides directly from a normative PDF via Ghostscript + Gemini 4 Hydrogen
app.post('/api/normative/process-batch', async (req: Request, res: Response) => {
  try {
    if (!ai) {
      return res.status(500).json({
        error: 'API Key de Gemini no configurada en el servidor.',
      });
    }

    const {
      deckId,
      deckTitle,
      pdfPath,
      totalSlides,
      startPage,
      endPage,
      generateAudio = false,
      voiceName = 'Kore',
      forceReprocess = false,
    } = req.body;

    if (!deckId || !pdfPath || !totalSlides || !startPage) {
      return res.status(400).json({
        error: 'Parámetros requeridos: deckId, pdfPath, totalSlides y startPage.',
      });
    }

    const safeDeckId = sanitizeIdPart(String(deckId));
    const maxPages = Math.max(1, Number(totalSlides));
    const fromPage = Math.max(1, Math.min(maxPages, Number(startPage)));
    const requestedEnd = endPage ? Number(endPage) : fromPage;
    // Clamp batch to max 5 slides per request for fast, reliable multimodal extraction
    const toPage = Math.max(fromPage, Math.min(maxPages, requestedEnd, fromPage + 4));

    const absPdfPath = resolveNormativePdfFile(String(pdfPath));
    if (!absPdfPath) {
      return res.status(404).json({
        error: 'No se encontró el archivo PDF solicitado en el servidor.',
      });
    }

    if (!normativeScriptsStore[safeDeckId]) {
      normativeScriptsStore[safeDeckId] = {};
    }

    const pagesToExtract: number[] = [];
    for (let p = fromPage; p <= toPage; p++) {
      if (forceReprocess || !normativeScriptsStore[safeDeckId][String(p)]) {
        pagesToExtract.push(p);
      }
    }

    const processedSlides: PersistedSlideScript[] = [];

    if (pagesToExtract.length > 0) {
      const minP = Math.min(...pagesToExtract);
      const maxP = Math.max(...pagesToExtract);
      const rendered = await renderPdfPagesToJpeg(absPdfPath, minP, maxP);
      const targetImages = rendered.filter((r) => pagesToExtract.includes(r.page));

      if (targetImages.length === 0) {
        return res.status(500).json({
          error: `No se pudieron renderizar las diapositivas ${minP}-${maxP} del PDF.`,
        });
      }

      const extracted = await extractStrictSlideBatch(
        String(deckTitle || safeDeckId),
        maxPages,
        targetImages
      );

      const nowIso = new Date().toISOString();
      for (const item of extracted) {
        const persisted: PersistedSlideScript = {
          page: item.page,
          title: item.title,
          keyTakeaway: item.keyTakeaway,
          narration: item.narration,
          verifiedAt: nowIso,
          source: 'pdf-batch',
        };
        normativeScriptsStore[safeDeckId][String(item.page)] = persisted;
      }
      saveScriptsStore(normativeScriptsStore);
    }

    for (let p = fromPage; p <= toPage; p++) {
      const saved = normativeScriptsStore[safeDeckId][String(p)];
      if (saved) processedSlides.push(saved);
    }

    const generatedAudioPages: number[] = [];
    const safeVoice = sanitizeVoiceName(voiceName);

    if (generateAudio) {
      for (const slide of processedSlides) {
        const slideKey = `${safeDeckId}_s${slide.page}`;
        const existing = !forceReprocess
          ? getPersistedSlideAudioIfFresh(slideKey, slide.narration, safeVoice)
          : null;
        if (existing) {
          generatedAudioPages.push(slide.page);
          continue;
        }
        try {
          const wavBuf = await synthesizeGeminiWav(slide.narration, safeVoice);
          persistNormativeSlideAudio(slideKey, slide.narration, safeVoice, wavBuf);
          generatedAudioPages.push(slide.page);
        } catch (audioErr) {
          console.error(`Error generando TTS para ${slideKey}:`, audioErr);
        }
      }
    }

    return res.json({
      deckId: safeDeckId,
      fromPage,
      toPage,
      slides: processedSlides,
      generatedAudioPages,
      audioIndex: normativeAudioIndex,
    });
  } catch (error: any) {
    return sendGeminiRouteError(
      res,
      '/api/normative/process-batch',
      error,
      'Error procesando el lote de diapositivas PDF.'
    );
  }
});

// Synthesize and persist TTS WAV for a single verified normative slide
app.post('/api/normative/synthesize-slide', async (req: Request, res: Response) => {
  try {
    if (!ai) {
      return res.status(500).json({
        error: 'API Key de Gemini no configurada en el servidor.',
      });
    }

    const { deckId, page, narration, voiceName = 'Kore', force = false } = req.body;
    if (!deckId || !page || !narration || typeof narration !== 'string') {
      return res.status(400).json({
        error: 'Se requiere deckId, page y narration para sintetizar el audio.',
      });
    }

    const safeDeckId = sanitizeIdPart(String(deckId));
    const pageNum = Number(page);
    const safeVoice = sanitizeVoiceName(voiceName);
    const slideKey = `${safeDeckId}_s${pageNum}`;

    if (!force) {
      const cached = getPersistedSlideAudioIfFresh(slideKey, narration, safeVoice);
      if (cached) {
        return res.json({
          ok: true,
          cached: true,
          slideKey,
          audioUrl: `/api/normative/tts-audio/${safeDeckId}/${pageNum}`,
          entry: normativeAudioIndex[slideKey],
        });
      }
    }

    const wavBuf = await synthesizeGeminiWav(narration, safeVoice);
    const entry = persistNormativeSlideAudio(slideKey, narration, safeVoice, wavBuf);

    return res.json({
      ok: true,
      cached: false,
      slideKey,
      audioUrl: `/api/normative/tts-audio/${safeDeckId}/${pageNum}`,
      entry,
    });
  } catch (error: any) {
    return sendGeminiRouteError(
      res,
      '/api/normative/synthesize-slide',
      error,
      'No se pudo sintetizar el audio WAV de esta diapositiva.'
    );
  }
});

// Serve or download a persisted normative slide TTS WAV file
app.get('/api/normative/tts-audio/:deckId/:page', (req: Request, res: Response) => {
  const safeDeckId = sanitizeIdPart(String(req.params.deckId || ''));
  const pageNum = parseInt(String(req.params.page || '0'), 10);
  if (!safeDeckId || Number.isNaN(pageNum) || pageNum < 1) {
    return res.status(400).json({ error: 'Identificador de audio inválido.' });
  }

  const slideKey = `${safeDeckId}_s${pageNum}`;
  const entry = normativeAudioIndex[slideKey];
  if (!entry) {
    return res.status(404).json({ error: 'Audio WAV aún no generado para esta diapositiva.' });
  }

  const fullPath = path.join(NORMATIVE_TTS_DIR, entry.filename);
  if (!fs.existsSync(fullPath)) {
    return res.status(404).json({ error: 'Archivo de audio WAV no encontrado en caché.' });
  }

  const stat = fs.statSync(fullPath);
  res.setHeader('Content-Type', 'audio/wav');
  res.setHeader('Content-Length', stat.size);
  res.setHeader('Cache-Control', 'public, max-age=86400');

  if (req.query.download === '1') {
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="${safeDeckId}_diapositiva_${String(pageNum).padStart(2, '0')}.wav"`
    );
  }

  return fs.createReadStream(fullPath).pipe(res);
});

// Visual slide reader & narrator endpoint (from rendered canvas JPEG) — upgraded to Gemini 4 Hydrogen + persists to disk
app.post('/api/ai/describe-slide', async (req: Request, res: Response) => {
  try {
    if (!ai) {
      return res.status(500).json({
        error: 'API Key de Gemini no configurada en el servidor.',
      });
    }

    const { imageBase64, deckId, deckTitle, pageNumber, totalSlides } = req.body;
    if (!imageBase64 || typeof imageBase64 !== 'string') {
      return res.status(400).json({ error: 'Imagen de diapositiva requerida.' });
    }

    const cleanedBase64 = imageBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');
    const pageNum = Math.max(1, Number(pageNumber || 1));
    const totalNum = Math.max(pageNum, Number(totalSlides || 1));

    const [extracted] = await extractStrictSlideBatch(
      String(deckTitle || 'Normativa Educativa'),
      totalNum,
      [{ page: pageNum, base64: cleanedBase64 }]
    );

    if (deckId) {
      const safeDeckId = sanitizeIdPart(String(deckId));
      if (!normativeScriptsStore[safeDeckId]) {
        normativeScriptsStore[safeDeckId] = {};
      }
      normativeScriptsStore[safeDeckId][String(pageNum)] = {
        page: pageNum,
        title: extracted.title,
        keyTakeaway: extracted.keyTakeaway,
        narration: extracted.narration,
        verifiedAt: new Date().toISOString(),
        source: 'canvas-visual',
      };
      saveScriptsStore(normativeScriptsStore);
    }

    return res.json(extracted);
  } catch (error: any) {
    return sendGeminiRouteError(
      res,
      '/api/ai/describe-slide',
      error,
      'No se pudo analizar visualmente esta diapositiva.'
    );
  }
});

// Infographic generation endpoint with Gemini 3 Pro Image Preview
app.post('/api/ai/infographic', async (req: Request, res: Response) => {
  try {
    if (!ai) {
      return res.status(500).json({
        error: 'API Key de Gemini no configurada.',
      });
    }

    const { topic, resolution = '1K' } = req.body;
    if (!topic) {
      return res.status(400).json({ error: 'Tema requerido para generar infografía.' });
    }

    const prompt = `Professional educational infographic diagram in Spanish about: "${topic}". Clean editorial academic design, elegant charts, clear pedagogical hierarchy, deep navy and warm cream palette, crisp typography, educational unit management framework.`;

    const response = await ai.models.generateImages({
      model: 'gemini-3-pro-image-preview',
      prompt,
      config: {
        numberOfImages: 1,
        aspectRatio: '16:9',
        imageSize: resolution as '1K' | '2K' | '4K',
      },
    });

    const base64Image = response.generatedImages?.[0]?.image?.imageBytes;
    if (base64Image) {
      return res.json({ imageBase64: `data:image/jpeg;base64,${base64Image}` });
    }

    return res.status(500).json({ error: 'No se pudo generar la imagen infográfica.' });
  } catch (error: any) {
    console.error('Error en /api/ai/infographic:', error);
    res.status(500).json({ error: error.message || 'Error generando infografía.' });
  }
});
// Helper to recursively add files to JSZip
function addDirectoryToZip(zip: JSZip, localDir: string, zipPrefix: string = '') {
  if (!fs.existsSync(localDir)) return;
  const items = fs.readdirSync(localDir);
  for (const item of items) {
    if (
      item === 'node_modules' ||
      item === 'dist' ||
      item === '.git' ||
      item === '.aistudio' ||
      item === '.DS_Store' ||
      item === 'bun.lock' ||
      item === 'normative-tts-cache' ||
      item.endsWith('.log')
    ) {
      continue;
    }
    const fullPath = path.join(localDir, item);
    const stat = fs.statSync(fullPath);
    const zipPath = zipPrefix ? `${zipPrefix}/${item}` : item;
    if (stat.isDirectory()) {
      addDirectoryToZip(zip, fullPath, zipPath);
    } else {
      const fileData = fs.readFileSync(fullPath);
      zip.file(zipPath, fileData);
    }
  }
}

// In-memory cache for generated zip buffers
let cachedAppZip: Buffer | null = null;
let cachedMaterialsZip: Buffer | null = null;

// Endpoint: Download full application (.ZIP)
app.get('/api/download/app-zip', async (_req: Request, res: Response) => {
  try {
    if (cachedAppZip) {
      res.set({
        'Content-Type': 'application/zip',
        'Content-Disposition': 'attachment; filename="director-pro-completo.zip"',
        'Content-Length': cachedAppZip.length,
        'Cache-Control': 'no-cache',
      });
      return res.send(cachedAppZip);
    }

    const zip = new JSZip();

    // 1. Add all project files
    addDirectoryToZip(zip, __dirname, '');

    // 2. Add LEEME_INSTRUCCIONES.md
    const readmeContent = `# DIRECTOR-PRO: Plataforma Preparatoria para Directores de Unidad Educativa
================================================================================
Examen de Institucionalización para Cargos Directivos del Sistema Educativo

¡Felicitaciones! Has descargado el paquete completo y autónomo de DIRECTOR-PRO.
Esta herramienta ha sido diseñada con el máximo rigor pedagógico para asegurar
un rendimiento sobresaliente en la evaluación directiva.

--------------------------------------------------------------------------------
🚀 INSTRUCCIONES PARA EJECUTAR EN TU COMPUTADORA (Windows, Mac o Linux)
--------------------------------------------------------------------------------

REQUISITOS PREVIOS:
- Tener instalado Node.js (versión 18 o superior).
  Si no lo tienes instalado, descárgalo gratis desde: https://nodejs.org

PASO A PASO:

A) EN WINDOWS:
   1. Haz doble clic en el archivo "INICIAR_WINDOWS.bat".
   2. El script instalará automáticamente las dependencias y levantará el sistema.
   3. Abre tu navegador (Google Chrome o Edge) en:
      👉 http://localhost:3000

B) EN MAC O LINUX:
   1. Abre una Terminal en esta carpeta.
   2. Ejecuta:
      npm install
      npm run dev
   3. Abre tu navegador en:
      👉 http://localhost:3000

--------------------------------------------------------------------------------
📦 CONTENIDO DE ESTE PAQUETE (26 FUENTES OFICIALES)
--------------------------------------------------------------------------------
1. BIBLIOTECA COMPLETA CON 26 FUENTES (12 DIRECTOR + 6 CATEDRÁTICO + 8 NORMATIVA GENERAL):
   - Director (Gestión): ODS 4, LIDEI 2022, Marco Gestión Local y ERCE 2019.
   - Director (Normativa Específica): R.M. 0001/2026, R.M. 0864/2019, Apertura/Cierre (0020/21/22-2024 y 0534/2025), R.M. 0190/2024, R.M. 0069/2025, R.M. 800/2015, D.S. 04688 y R.S. 212414.
   - Catedrático: Mora, UNIR, Barraza, D.S. 04688, Compendio ESFM 2018 y R.S. 212414.
   - Normativa General: CPE y Leyes 070, 045, 348, 548, 223, 1178 (SAFCO) y 004.

2. LOS 7 GUIONES LIMPIOS DE ESTUDIO AUDITIVO (archivos .txt en "public/assets/aistudio/"):
   - 01_LIDEI_2022_Guion_Limpio_Google_Studio_da4d2280.txt
   - 02_Marco_Gestion_Liderazgo_Guion_Limpio_Google_Studio__1__31f9f86d.txt
   - 03_ERCE_2019_Guion_Limpio_Google_Studio_b5fd0332.txt
   - 04_ODS4_Guion_Limpio_Google_Studio_4295119d.txt
   - Guion_Limpio_Neuroeducacion_Mora_Para_Google_AI_Studio_87a634bb.txt
   - Guion_Limpio_Didactica_General_Google_AI_Studio_b79c0e5c.txt
   - Guion_Limpio_Investigacion_Cualitativa_Barraza_Google_AI_Studio_9dc2c5e7.txt

3. 11 MÓDULOS DE ESTUDIO TÉCNICO-PEDAGÓGICO:
   - 1. Biblioteca digital con lector buscable completo y visor PDF.
   - 2. Resúmenes Analíticos exhaustivos de las 26 fuentes.
   - 3. Presentaciones Didácticas completas con notas para el orador y diapositivas.
   - 4. Mapas Conceptuales interactivos y visuales (26 árboles).
   - 5. Estudio Auditivo con síntesis de voz (TTS - 26 cápsulas).
   - 6. Audios Oficiales estructurados bloque por bloque (26 sesiones).
   - 7. Casos Prácticos con 30 dilemas situacionales reales.
   - 8. Simulador de Examen con 1.300 preguntas oficiales (simulacros de 50 preguntas por libro).
   - 9. Guías Didácticas de Estudio (26 guías integrales).
   - 10. Preguntas y Respuestas por Libro con referencia corta física (1.300 reactivos).
   - 11. Diapositivas Técnico-Pedagógicas (5 presentaciones ilustradas en PDF: 198 diapositivas).

4. MODO OFFLINE / SIN INTERNET:
   - Todos los bancos de preguntas, resúmenes, presentaciones, casos, mapas conceptuales
     y lectura de guiones funcionan al 100% de manera autónoma sin conexión a internet.
   - Opcional: Si deseas activar el Consultor en Vivo con IA, puedes crear un archivo ".env"
     en la carpeta con tu GEMINI_API_KEY.

¡Te deseamos el mayor de los éxitos en tu examen de institucionalización directiva!
`;
    zip.file('LEEME_INSTRUCCIONES.md', readmeContent);

    // 3. Add INICIAR_WINDOWS.bat
    const windowsBatContent = `@echo off
title DIRECTOR-PRO - Preparador Institucional para Directores
echo ========================================================
echo   DIRECTOR-PRO: Preparador Institucional para Directores
echo ========================================================
echo.
echo [1/2] Verificando dependencias...
call npm install
echo.
echo [2/2] Iniciando aplicacion en puerto local 3000...
start http://localhost:3000
call npm run dev
pause
`;
    zip.file('INICIAR_WINDOWS.bat', windowsBatContent);

    // 4. Add iniciar_mac_linux.sh
    const bashScriptContent = `#!/usr/bin/env bash
echo "========================================================"
echo "  DIRECTOR-PRO: Preparador Institucional para Directores"
echo "========================================================"
echo "[1/2] Instalando dependencias..."
npm install
echo "[2/2] Iniciando servidor..."
echo "Abre tu navegador en: http://localhost:3000"
npm run dev
`;
    zip.file('iniciar_mac_linux.sh', bashScriptContent);

    // Generate zip buffer
    const zipBuffer = await zip.generateAsync({
      type: 'nodebuffer',
      compression: 'DEFLATE',
      compressionOptions: { level: 6 },
    });

    cachedAppZip = zipBuffer;

    res.set({
      'Content-Type': 'application/zip',
      'Content-Disposition': 'attachment; filename="director-pro-completo.zip"',
      'Content-Length': zipBuffer.length,
      'Cache-Control': 'no-cache',
    });
    return res.send(zipBuffer);
  } catch (error: any) {
    console.error('Error generando zip de la aplicación:', error);
    res.status(500).json({ error: 'No se pudo generar el paquete zip de la aplicación.' });
  }
});

// Endpoint: Download study materials bundle (PDFs + TXT Guiones)
app.get('/api/download/materials-bundle', async (_req: Request, res: Response) => {
  try {
    if (cachedMaterialsZip) {
      res.set({
        'Content-Type': 'application/zip',
        'Content-Disposition': 'attachment; filename="bibliografia-y-guiones-director-pro.zip"',
        'Content-Length': cachedMaterialsZip.length,
        'Cache-Control': 'no-cache',
      });
      return res.send(cachedMaterialsZip);
    }

    const zip = new JSZip();
    const assetsDir = path.join(__dirname, 'public', 'assets', 'aistudio');

    if (fs.existsSync(assetsDir)) {
      const files = fs.readdirSync(assetsDir);
      for (const file of files) {
        const filePath = path.join(assetsDir, file);
        const data = fs.readFileSync(filePath);
        zip.file(file, data);
      }
    }

    const guideContent = `# BIBLIOGRAFÍA OFICIAL Y GUIONES TÉCNICO-PEDAGÓGICOS
DIRECTOR & CATEDRÁTICO PRO - Plataforma de Alta Exigencia 2026

Este paquete contiene los documentos oficiales originales en formato PDF disponibles,
el índice completo de las 26 fuentes oficiales integradas en la plataforma y los
guiones limpios redactados para el estudio auditivo:

=== BIBLIOGRAFÍA PARA DIRECTOR DE UNIDAD EDUCATIVA (4 GESTIÓN + 8 NORMATIVA ESPECÍFICA) ===
1. 2.1_ODS_4_UNESCO_2030_4799493d.pdf (ODS 4 - Declaración de Incheon / UNESCO 2030)
2. 2.2_Lidei_2022_Liderazgo_Ed_OEI_f1367d9d.pdf (LIDEI 2022 - OEI Liderazgo Escolar)
3. 2.3_Marco_Gesti_n_Intermedia_754e2b67.pdf (Marco para la Gestión y Liderazgo Local - Ley 21.040)
4. 2.4_Estudio_Regional_ERCE_2019_e11e6fc3.pdf (ERCE 2019 - UNESCO LLECE)
5. R.M. N° 0001/2026 - Normas Generales para la Gestión Educativa 2026 (Regular) [Integrado en Lector/Guías]
6. R.M. N° 0864/2019 - Protocolo de Violencia Física, Psicológica y Sexual en UEs y CEE [Integrado en Lector/Guías]
7. Compilado Apertura, Modificación y Cierre de UEs (R.M. 0020, 0021, 0022/2024 y R.M. 0534/2025) [Integrado en Lector/Guías]
8. R.M. N° 0190/2024 - Reglamento de Evaluación del Desarrollo Curricular RE-EDC V.01 [Integrado en Lector/Guías]
9. R.M. N° 0069/2025 - Fortalecimiento de Educación Inclusiva Modalidad Indirecta RPFEISEP-MI [Integrado en Lector/Guías]
10. R.M. N° 800/2015 - Libretas y Certificados Electrónicos Alternativa (SIEALT-MN-070-400) [Integrado en Lector/Guías]
11. 9.-_DS._04688_b0dc5656.pdf (D.S. N° 04688 - Escalafón Nacional, Enfoque Director)
12. 11._Resoluci_n_Suprema_N__212414_872fdfc9.pdf (R.S. N° 212414 - Faltas y Sanciones, Enfoque Director)

=== BIBLIOGRAFÍA PARA DOCENTE CATEDRÁTICO (TÉCNICO-PEDAGÓGICO Y NORMATIVA ESPECÍFICA) ===
13. 1._Neuroeducacion_-_solo_se_puede_aprender_aquello_que_se_ama_ba78867a.pdf (Neuroeducación - Francisco Mora / Alianza Editorial)
14. LOPEZ_GOMEZ_DIDACTICA_GENERAL_Y_FORMACION_DEL_PROFESORADO_EDITORIAL_UNIR_38ee6f77.pdf (Didáctica General y Formación del Profesorado - Ernesto López Gómez / UNIR)
15. 3._Metodologia_de_la_Investigacion_cualitativa_640a256f.pdf (Metodología de la Investigación Cualitativa Interpretativa - Arturo Barraza Macías)
16. 9.-_DS._04688_b0dc5656.pdf (D.S. N° 04688 - Reglamento del Escalafón Nacional del Servicio de Educación)
17. 10._Compedio_Normativo_para_las_ESFM_db7caffc.pdf (Compendio de Normativa para las ESFM y Unidades Académicas - Versión 2018)
18. 11._Resoluci_n_Suprema_N__212414_872fdfc9.pdf (R.S. N° 212414 - Reglamento de Faltas y Sanciones del Magisterio)

=== DOCUMENTOS DE NORMATIVA GENERAL (BOLIVIA - COMÚN PARA AMBOS CARGOS) ===
19. 1._CPE_Bolivia_1f28d9e8.pdf (Constitución Política del Estado Plurinacional de Bolivia)
20. 2._Ley_070_faf45b8c.pdf (Ley N° 070 de la Educación "Avelino Siñani - Elizardo Pérez")
21. 3._Ley_N__045_948d566f.pdf (Ley N° 045 Contra el Racismo y Toda Forma de Discriminación)
22. 4.-_Ley_N__348_b474bc9d.pdf (Ley N° 348 Integral para Garantizar a las Mujeres una Vida Libre de Violencia)
23. 5.-_Ley_N__548_f31af2f9.pdf (Ley N° 548 Código Niña, Niño y Adolescente)
24. 6.-_Ley_N__223_13e90177.pdf (Ley N° 223 Ley General para Personas con Discapacidad)
25. 7.-_Ley_N__1178_9cda8962.pdf (Ley N° 1178 de Administración y Control Gubernamentales - SAFCO)
26. 8.-_Ley_N__004_ae2c5071.pdf (Ley N° 004 de Lucha Contra la Corrupción Marcelo Quiroga Santa Cruz)

=== GUIONES DE AUDIO LIMPIOS ===
- 01_LIDEI_2022_Guion_Limpio_Google_Studio_da4d2280.txt (LIDEI - Liderazgo Escolar)
- 02_Marco_Gestion_Liderazgo_Guion_Limpio_Google_Studio__1__31f9f86d.txt (Marco de Gestión)
- 03_ERCE_2019_Guion_Limpio_Google_Studio_b5fd0332.txt (ERCE 2019)
- 04_ODS4_Guion_Limpio_Google_Studio_4295119d.txt (ODS 4 UNESCO 2030)
- Guion_Limpio_Neuroeducacion_Mora_Para_Google_AI_Studio_87a634bb.txt (Neuroeducación - Mora)
- Guion_Limpio_Didactica_General_Google_AI_Studio_b79c0e5c.txt (Didáctica General - UNIR)
- Guion_Limpio_Investigacion_Cualitativa_Barraza_Google_AI_Studio_9dc2c5e7.txt (Inv. Cualitativa - Barraza)

=== 26 GUÍAS DIDÁCTICAS OFICIALES DE ESTUDIO ===
1-4. Guías Dirección Gestión: ODS 4, LIDEI OEI 2022, Marco Gestión Local, ERCE 2019
5-12. Guías Dirección Normativa Específica: R.M. 0001/2026, R.M. 0864/2019, Apertura R.M. 0020-0534, R.M. 0190/2024, R.M. 0069/2025, R.M. 800/2015, D.S. 04688, R.S. 212414
13-18. Guías Cátedra: Neuroeducación, Didáctica General, Inv. Cualitativa, D.S. 04688, Compendio ESFM 2018, R.S. 212414
19-26. Guías Normativa General: CPE, Ley 070, Ley 045, Ley 348, Ley 548, Ley 223, Ley 1178, Ley 004

=== 5 PRESENTACIONES ILUSTRADAS DE DIAPOSITIVAS TÉCNICO-PEDAGÓGICAS (198 DIAPOSITIVAS) ===
- 2.1._Marco_de_Acci_n_ODS_4_completo_a0ebbcf2.pdf (53 diapositivas)
- LIDEI_INTRO_ULTIMO_3079d708.pdf (32 diapositivas)
- 2.2._Liderazgo_Directivo_en_Iberoam_rica_2949203e.pdf (40 diapositivas)
- 2.3._Liderazgo_Educativo_Local__1__490e4048.pdf (41 diapositivas)
- 2.4._Aprendizajes_Fundamentales_ERCE_2019COMPLETO_6a3151e9.pdf (32 diapositivas)
`;
    zip.file('INDICE_DOCUMENTOS.txt', guideContent);

    const zipBuffer = await zip.generateAsync({
      type: 'nodebuffer',
      compression: 'DEFLATE',
      compressionOptions: { level: 6 },
    });

    cachedMaterialsZip = zipBuffer;

    res.set({
      'Content-Type': 'application/zip',
      'Content-Disposition': 'attachment; filename="bibliografia-y-guiones-director-pro.zip"',
      'Content-Length': zipBuffer.length,
      'Cache-Control': 'no-cache',
    });
    return res.send(zipBuffer);
  } catch (error: any) {
    console.error('Error generando zip de materiales:', error);
    res.status(500).json({ error: 'No se pudo generar el paquete de materiales.' });
  }
});

// Endpoint: Download status & info
app.get('/api/download/info', (_req: Request, res: Response) => {
  res.json({
    appName: 'DIRECTOR & CATEDRÁTICO PRO',
    version: '5.0.0',
    fullPackageName: 'director-y-catedratico-pro.zip',
    materialsPackageName: 'bibliografia-oficial-26-fuentes.zip',
    estimatedSizeMb: 34.0,
    pdfCount: 26,
    scriptCount: 7,
    questionCount: 1300,
    modulesCount: 11,
    pwaSupported: true,
  });
});

// Dedicated Range-capable streaming handler for large PDF/media assets (>32 MB safe for Cloud Run)
const MAX_RANGE_CHUNK_BYTES = 4 * 1024 * 1024; // 4 MB max per 206 chunk

function getAssetMimeType(filename: string): string {
  const ext = path.extname(filename).toLowerCase();
  switch (ext) {
    case '.pdf':
      return 'application/pdf';
    case '.txt':
      return 'text/plain; charset=utf-8';
    case '.mp3':
      return 'audio/mpeg';
    case '.wav':
      return 'audio/wav';
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.webp':
      return 'image/webp';
    default:
      return 'application/octet-stream';
  }
}

app.use('/assets/aistudio', (req: Request, res: Response, next) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    return next();
  }

  let decodedRelPath: string;
  try {
    decodedRelPath = decodeURIComponent(req.path).replace(/^\/+/, '');
  } catch {
    return res.status(400).json({ error: 'Ruta de archivo inválida.' });
  }

  if (!decodedRelPath || decodedRelPath.includes('..') || decodedRelPath.includes('\0')) {
    return res.status(400).json({ error: 'Ruta de archivo no permitida.' });
  }

  const candidateDirs = [
    path.join(__dirname, 'public', 'assets', 'aistudio'),
    path.join(__dirname, 'dist', 'assets', 'aistudio'),
  ];

  let resolvedFilePath: string | null = null;
  for (const baseDir of candidateDirs) {
    const normalizedBase = path.resolve(baseDir);
    const candidatePath = path.resolve(normalizedBase, decodedRelPath);
    if (
      candidatePath.startsWith(normalizedBase + path.sep) &&
      fs.existsSync(candidatePath) &&
      fs.statSync(candidatePath).isFile()
    ) {
      resolvedFilePath = candidatePath;
      break;
    }
  }

  if (!resolvedFilePath) {
    return next();
  }

  const stat = fs.statSync(resolvedFilePath);
  const totalSize = stat.size;
  const contentType = getAssetMimeType(resolvedFilePath);

  res.setHeader('Content-Type', contentType);
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Cache-Control', 'public, max-age=3600');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Access-Control-Expose-Headers', 'Accept-Ranges, Content-Range, Content-Length');

  if (req.method === 'HEAD') {
    res.setHeader('Content-Length', totalSize);
    return res.status(200).end();
  }

  const rangeHeader = req.headers.range;
  if (rangeHeader) {
    const match = /^bytes=(\d*)-(\d*)$/i.exec(rangeHeader.trim());
    if (!match) {
      res.setHeader('Content-Range', `bytes */${totalSize}`);
      return res.status(416).end();
    }

    let start: number;
    let end: number;

    if (match[1] === '' && match[2] !== '') {
      // Suffix range: bytes=-N
      const suffixLength = parseInt(match[2], 10);
      if (Number.isNaN(suffixLength) || suffixLength <= 0) {
        res.setHeader('Content-Range', `bytes */${totalSize}`);
        return res.status(416).end();
      }
      start = Math.max(0, totalSize - suffixLength);
      end = totalSize - 1;
    } else {
      start = parseInt(match[1], 10);
      end = match[2] !== '' ? parseInt(match[2], 10) : totalSize - 1;
    }

    if (Number.isNaN(start) || Number.isNaN(end) || start < 0 || start >= totalSize || start > end) {
      res.setHeader('Content-Range', `bytes */${totalSize}`);
      return res.status(416).end();
    }

    // Clamp single 206 response chunk size so Cloud Run's 32 MB limit is never exceeded
    end = Math.min(end, totalSize - 1, start + MAX_RANGE_CHUNK_BYTES - 1);
    const chunkSize = end - start + 1;

    res.status(206);
    res.setHeader('Content-Range', `bytes ${start}-${end}/${totalSize}`);
    res.setHeader('Content-Length', chunkSize);

    const stream = fs.createReadStream(resolvedFilePath, { start, end });
    stream.on('error', (err) => {
      console.error('Error streaming PDF range:', err);
      if (!res.headersSent) {
        res.status(500).end();
      } else {
        res.end();
      }
    });
    res.on('close', () => {
      if (!stream.destroyed) {
        stream.destroy();
      }
    });
    return stream.pipe(res);
  }

  // Non-ranged GET: stream directly (small/medium files under Cloud Run limit)
  res.setHeader('Content-Length', totalSize);
  const fullStream = fs.createReadStream(resolvedFilePath);
  fullStream.on('error', (err) => {
    console.error('Error streaming asset file:', err);
    if (!res.headersSent) {
      res.status(500).end();
    } else {
      res.end();
    }
  });
  res.on('close', () => {
    if (!fullStream.destroyed) {
      fullStream.destroy();
    }
  });
  return fullStream.pipe(res);
});

// Static or Vite setup
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(
      express.static(path.join(__dirname, 'dist'), {
        setHeaders: (res, filePath) => {
          if (filePath.endsWith('.html') || filePath.endsWith('manifest.json')) {
            res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
            res.setHeader('Pragma', 'no-cache');
            res.setHeader('Expires', '0');
          }
        },
      })
    );
    app.get('*', (_req, res) => {
      res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use((req, res, next) => {
      const isHtmlDoc =
        req.method === 'GET' &&
        (req.path === '/' ||
          req.path.endsWith('.html') ||
          req.path === '/manifest.json' ||
          (!req.path.startsWith('/api/') &&
            !req.path.startsWith('/assets/') &&
            !req.path.startsWith('/src/') &&
            !req.path.startsWith('/@') &&
            !req.path.includes('.')));
      if (isHtmlDoc) {
        const origSetHeader = res.setHeader.bind(res);
        res.setHeader = ((name: string, value: any) => {
          if (name.toLowerCase() === 'cache-control') {
            return origSetHeader(
              'Cache-Control',
              'no-store, no-cache, must-revalidate, proxy-revalidate'
            );
          }
          if (name.toLowerCase() === 'etag') {
            return res;
          }
          return origSetHeader(name, value);
        }) as typeof res.setHeader;
        origSetHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
        origSetHeader('Pragma', 'no-cache');
        origSetHeader('Expires', '0');
      }
      next();
    });
    app.use(vite.middlewares);
  }

  app.listen(port, () => {
    console.log(`Servidor DIRECTOR-PRO corriendo en http://localhost:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Fallo al iniciar el servidor:', err);
});
