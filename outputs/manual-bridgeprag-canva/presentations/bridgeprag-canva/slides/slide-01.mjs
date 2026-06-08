const C = {
  blue: "#2458ff",
  blue2: "#0ea5ff",
  red: "#ff3030",
  green: "#198c31",
  purple: "#7046ff",
  navy: "#18233f",
  gray: "#8a94a8",
  light: "#ffffff",
  panel: "#fbfcff",
}

function text(slide, ctx, value, x, y, w, h, opts = {}) {
  return ctx.addText(slide, {
    text: value,
    left: x,
    top: y,
    width: w,
    height: h,
    fontSize: opts.size ?? 18,
    color: opts.color ?? C.navy,
    bold: opts.bold ?? false,
    align: opts.align ?? "center",
    valign: opts.valign ?? "middle",
    typeface: opts.face ?? "Aptos",
    fill: opts.fill ?? "#00000000",
    line: opts.line ?? ctx.line(),
    insets: opts.insets ?? { left: 4, right: 4, top: 2, bottom: 2 },
  })
}

function box(slide, ctx, x, y, w, h, color, fill = C.panel) {
  return ctx.addShape(slide, {
    left: x,
    top: y,
    width: w,
    height: h,
    geometry: "roundRect",
    fill,
    line: ctx.line(color, 2),
  })
}

function line(slide, ctx, x1, y1, x2, y2, color = C.navy, width = 2, style = "solid") {
  return ctx.addShape(slide, {
    left: x1,
    top: y1,
    width: x2 - x1,
    height: y2 - y1,
    geometry: "line",
    fill: "#00000000",
    line: ctx.line(color, width, style),
  })
}

function arrowRight(slide, ctx, x1, y, x2, color = C.navy, width = 2) {
  line(slide, ctx, x1, y, x2 - 10, y, color, width)
  text(slide, ctx, ">", x2 - 15, y - 13, 16, 26, { size: 18, color, bold: true })
}

function smallBar(slide, ctx, x, y, color) {
  ctx.addShape(slide, {
    left: x,
    top: y,
    width: 16,
    height: 16,
    geometry: "rect",
    fill: `${color}18`,
    line: ctx.line(color, 1.6),
  })
}

function memoryColumn(slide, ctx, x, y, color, count = 8) {
  for (let i = 0; i < count; i += 1) smallBar(slide, ctx, x, y + i * 17, color)
}

function mlp(slide, ctx, x, y) {
  const layers = [
    [x, y + 52, y + 112, y + 172],
    [x + 60, y + 22, y + 82, y + 142, y + 202],
    [x + 122, y + 62, y + 122, y + 182],
  ]
  const points = []
  layers.forEach((layer, li) => {
    for (let i = 1; i < layer.length; i += 1) {
      points.push({ x: layer[0], y: layer[i], layer: li })
    }
  })
  for (const a of points.filter((p) => p.layer === 0)) {
    for (const b of points.filter((p) => p.layer === 1)) line(slide, ctx, a.x + 10, a.y, b.x - 10, b.y, C.green, 1)
  }
  for (const a of points.filter((p) => p.layer === 1)) {
    for (const b of points.filter((p) => p.layer === 2)) line(slide, ctx, a.x + 10, a.y, b.x - 10, b.y, C.green, 1)
  }
  points.forEach((p) => {
    ctx.addShape(slide, {
      left: p.x - 10,
      top: p.y - 10,
      width: 20,
      height: 20,
      geometry: "ellipse",
      fill: "#ffffff",
      line: ctx.line(C.green, 2),
    })
  })
}

function adapterBlock(slide, ctx, x, y) {
  box(slide, ctx, x, y, 70, 54, C.red, "#ffffff")
  text(slide, ctx, "Adapter\n(Linear)", x + 4, y + 9, 62, 36, { size: 11, color: C.navy, bold: false })
}

function arrowDown(slide, ctx, x, y1, y2, color = C.navy, width = 1.8) {
  line(slide, ctx, x, y1, x, y2 - 10, color, width)
  text(slide, ctx, "v", x - 8, y2 - 18, 16, 18, { size: 15, color, bold: true })
}

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add()

  ctx.addShape(slide, {
    left: 0,
    top: 0,
    width: ctx.W,
    height: ctx.H,
    geometry: "rect",
    fill: "#ffffff",
    line: ctx.line("#ffffff", 0),
  })

  const panels = {
    input: [16, 28, 156, 520, C.blue],
    mlp: [190, 28, 196, 520, C.green],
    kv: [410, 28, 288, 520, C.red],
    merge: [720, 28, 288, 520, C.blue2],
    llm: [1030, 28, 180, 520, C.purple],
    answer: [1222, 272, 50, 116, C.blue],
  }
  for (const [x, y, w, h, color] of Object.values(panels)) box(slide, ctx, x, y, w, h, color, "#ffffff")

  text(slide, ctx, "Input", 40, 56, 108, 24, { size: 16, color: C.blue, bold: true })
  text(slide, ctx, "Q + P", 40, 82, 108, 24, { size: 16, color: C.navy, bold: true, face: "Aptos Italic" })
  text(slide, ctx, "Question (Q)", 36, 126, 116, 20, { size: 11, color: C.blue, bold: true })
  box(slide, ctx, 48, 150, 76, 62, C.gray, "#ffffff")
  text(slide, ctx, "?", 54, 158, 38, 38, { size: 32, color: C.blue, bold: true })
  line(slide, ctx, 88, 172, 112, 172, C.gray, 1.6)
  line(slide, ctx, 88, 188, 112, 188, C.gray, 1.6)
  text(slide, ctx, "+", 52, 250, 80, 54, { size: 34, color: C.blue, bold: true })
  text(slide, ctx, "Passage (P)", 36, 344, 116, 20, { size: 11, color: C.blue, bold: true })
  box(slide, ctx, 48, 380, 76, 62, C.gray, "#ffffff")
  text(slide, ctx, "P", 70, 392, 32, 32, { size: 22, color: C.gray, bold: true })
  line(slide, ctx, 88, 420, 112, 420, C.gray, 1.6)

  text(slide, ctx, "MLP", 216, 56, 144, 24, { size: 18, color: C.green, bold: true })
  text(slide, ctx, "(Neural Network)", 216, 82, 144, 20, { size: 12, color: C.navy })
  mlp(slide, ctx, 228, 214)

  arrowRight(slide, ctx, 392, 286, 410, C.navy, 2)

  text(slide, ctx, "K/V Memory Generation", 446, 54, 218, 24, { size: 17, color: C.red, bold: true })
  text(slide, ctx, "(with Adapter)", 486, 80, 140, 22, { size: 14, color: C.red, bold: true })
  text(slide, ctx, "K Memory", 430, 124, 88, 18, { size: 11, color: C.navy, align: "left" })
  memoryColumn(slide, ctx, 450, 150, C.purple, 8)
  arrowRight(slide, ctx, 475, 218, 532, C.navy, 1.6)
  adapterBlock(slide, ctx, 535, 192)
  arrowRight(slide, ctx, 610, 218, 656, C.navy, 1.6)
  memoryColumn(slide, ctx, 656, 150, C.purple, 8)
  text(slide, ctx, "K", 680, 216, 26, 24, { size: 18, color: C.purple, bold: true, face: "Aptos Italic" })
  line(slide, ctx, 430, 330, 680, 330, C.purple, 1.2, "dashed")
  text(slide, ctx, "V Memory", 430, 366, 88, 18, { size: 11, color: C.navy, align: "left" })
  memoryColumn(slide, ctx, 450, 392, C.blue2, 8)
  arrowRight(slide, ctx, 475, 460, 532, C.navy, 1.6)
  adapterBlock(slide, ctx, 535, 434)
  arrowRight(slide, ctx, 610, 460, 656, C.navy, 1.6)
  memoryColumn(slide, ctx, 656, 392, C.blue2, 8)
  text(slide, ctx, "V", 680, 458, 26, 24, { size: 18, color: C.blue2, bold: true, face: "Aptos Italic" })

  arrowRight(slide, ctx, 698, 286, 720, C.navy, 2)

  text(slide, ctx, "Orthogonal Merge", 760, 54, 210, 24, { size: 17, color: "#075fb4", bold: true })
  text(slide, ctx, "(Merge into One K*, V* Memory)", 748, 80, 236, 22, { size: 12, color: "#075fb4", bold: true })
  text(slide, ctx, "K vectors (many)", 744, 126, 120, 18, { size: 10, color: C.purple, face: "Aptos Italic", align: "left" })
  ;[148, 188, 228, 268].forEach((yy) => {
    ctx.addShape(slide, { left: 748, top: yy, width: 58, height: 16, geometry: "rect", fill: "#7c3aed18", line: ctx.line(C.purple, 1.4) })
    line(slide, ctx, 806, yy + 8, 885, 238, C.purple, 1.2, "dashed")
  })
  line(slide, ctx, 885, 154, 885, 312, C.navy, 1.8)
  line(slide, ctx, 885, 312, 930, 312, C.navy, 1.8)
  ctx.addShape(slide, { left: 930, top: 304, width: 50, height: 16, geometry: "rect", fill: "#7c3aed18", line: ctx.line(C.purple, 1.4) })
  text(slide, ctx, "K*", 982, 298, 32, 26, { size: 16, color: C.purple, bold: true, face: "Aptos Italic" })
  line(slide, ctx, 740, 330, 990, 330, C.purple, 1.2, "dashed")
  text(slide, ctx, "V vectors (many)", 744, 366, 120, 18, { size: 10, color: C.blue2, face: "Aptos Italic", align: "left" })
  ;[390, 430, 470, 510].forEach((yy) => {
    ctx.addShape(slide, { left: 748, top: yy, width: 58, height: 16, geometry: "rect", fill: "#0ea5ff18", line: ctx.line(C.blue2, 1.4) })
    line(slide, ctx, 806, yy + 8, 885, 480, C.blue2, 1.2, "dashed")
  })
  line(slide, ctx, 885, 394, 885, 520, C.navy, 1.8)
  line(slide, ctx, 885, 520, 930, 520, C.navy, 1.8)
  ctx.addShape(slide, { left: 930, top: 512, width: 50, height: 16, geometry: "rect", fill: "#0ea5ff18", line: ctx.line(C.blue2, 1.4) })
  text(slide, ctx, "V*", 982, 506, 32, 26, { size: 16, color: C.blue2, bold: true, face: "Aptos Italic" })

  text(slide, ctx, "Inject into\nFrozen LLM", 1058, 54, 122, 54, { size: 14, color: C.navy, bold: true })
  text(slide, ctx, "Frozen LLM", 1082, 128, 104, 24, { size: 13, color: C.blue, bold: true })
  text(slide, ctx, "*", 1050, 122, 32, 32, { size: 28, color: C.blue, bold: true })
  box(slide, ctx, 1060, 190, 120, 54, C.gray, "#ffffff")
  text(slide, ctx, "Layer l + 1", 1070, 204, 100, 22, { size: 13, color: C.navy, face: "Aptos Italic" })
  arrowDown(slide, ctx, 1120, 250, 292, C.gray, 1.6)
  box(slide, ctx, 1060, 292, 120, 62, C.blue, "#ffffff")
  text(slide, ctx, "Layer l\n(Attention)", 1070, 304, 100, 34, { size: 13, color: "#075fb4", bold: true, face: "Aptos Italic" })
  arrowDown(slide, ctx, 1120, 360, 400, C.gray, 1.6)
  box(slide, ctx, 1060, 400, 120, 54, C.gray, "#ffffff")
  text(slide, ctx, "Layer l - 1", 1070, 414, 100, 22, { size: 13, color: C.navy, face: "Aptos Italic" })

  line(slide, ctx, 1010, 312, 1030, 312, C.purple, 2)
  line(slide, ctx, 1030, 312, 1060, 312, C.purple, 2)
  line(slide, ctx, 1010, 520, 1030, 520, C.blue2, 2)
  line(slide, ctx, 1030, 332, 1030, 520, C.blue2, 2)
  line(slide, ctx, 1030, 332, 1060, 332, C.blue2, 2)
  arrowRight(slide, ctx, 1180, 323, 1222, C.navy, 2)

  text(slide, ctx, "Answer", 1228, 292, 40, 18, { size: 8, color: C.blue, bold: true })
  box(slide, ctx, 1238, 318, 26, 38, C.blue, "#ffffff")
  line(slide, ctx, 1244, 338, 1258, 338, C.blue, 1.3)
  line(slide, ctx, 1244, 348, 1258, 348, C.blue, 1.3)

  return slide
}
