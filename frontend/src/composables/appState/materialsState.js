import { ref } from 'vue'

export function useMaterialsState({
  fileTree,
  activeFileId,
  activeFileName,
  currentAttachments,
  ensureLectureOneFile,
  updateNodeById
}) {
  const currentPreviewMaterial = ref(null)

  const handleUploadLectureMaterials = (files) => {
    const file = files[0]
    if (!file) return

    const uploadedAt = new Date().toISOString()
    const nextAttachment = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      name: file.name,
      size: file.size,
      type: file.type,
      uploadedAt,
      url: URL.createObjectURL(file),
      sourceFile: file
    }

    fileTree.value = updateNodeById(
      ensureLectureOneFile(fileTree.value),
      'lecture-1',
      (node) => ({
        ...node,
        attachments: [nextAttachment, ...(node.attachments || [])]
      })
    )

    currentPreviewMaterial.value = nextAttachment
    activeFileId.value = 'lecture-1'
    activeFileName.value = '강의1'
  }

  const handleClosePreviewMaterial = () => {
    currentPreviewMaterial.value = null
  }

  const handleOpenStoredMaterial = (materialId) => {
    const target = currentAttachments.value.find((item) => item.id === materialId)
    if (!target) return
    currentPreviewMaterial.value = target
  }

  const handleDeleteStoredMaterial = (materialId) => {
    fileTree.value = updateNodeById(
      ensureLectureOneFile(fileTree.value),
      activeFileId.value,
      (node) => ({
        ...node,
        attachments: (node.attachments || []).filter((item) => item.id !== materialId)
      })
    )

    if (currentPreviewMaterial.value?.id === materialId) {
      currentPreviewMaterial.value = null
    }
  }

  return {
    currentPreviewMaterial,
    handleUploadLectureMaterials,
    handleClosePreviewMaterial,
    handleOpenStoredMaterial,
    handleDeleteStoredMaterial
  }
}
