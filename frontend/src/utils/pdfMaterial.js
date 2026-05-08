export const isPdfMaterial = (material = {}) => {
  const name = String(material?.name || material?.title || '')
  const type = String(material?.type || material?.fileType || material?.mimeType || '')
  const url = String(material?.url || '')

  return type.toLowerCase().includes('pdf') || /\.pdf(?:$|\?)/i.test(name) || /\.pdf(?:$|\?)/i.test(url)
}
