import { ref, watch } from 'vue'
import { isWorkspaceUuid, saveSessionResources, uploadWorkspaceMaterial } from '../../api/workspaceApi.js'
import { addMaterialToCurrentWeek, findNodeById } from './fileTreeState'

// 워크스페이스에 올린 PDF/PPT 강의자료 첨부와 현재 미리보기 자료를 관리합니다.
export function useMaterialsState({
  fileTree,
  activeFileId,
  activeFileName,
  currentAttachments,
  normalizeFileTree,
  updateNodeById
}) {
  const currentPreviewMaterial = ref(null)

  watch(currentAttachments, (nextAttachments) => {
    if (!currentPreviewMaterial.value) return

    const nextMaterial = nextAttachments.find((item) => item.id === currentPreviewMaterial.value.id)
    currentPreviewMaterial.value = nextMaterial || null
  })

  // 선택한 파일을 현재 작업 파일의 이번 주차 강의자료 폴더에 추가하고 바로 미리보기로 엽니다.
  const handleUploadLectureMaterials = async (files) => {
    const file = files[0]
    if (!file) return

    const targetFileId = activeFileId.value
    if (!targetFileId) return

    const nextAttachment = isWorkspaceUuid(targetFileId)
      ? await uploadWorkspaceMaterial(targetFileId, file)
      : {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          name: file.name,
          size: file.size,
          type: file.type,
          uploadedAt: new Date().toISOString(),
          url: URL.createObjectURL(file),
          sourceFile: file
        }

    fileTree.value = updateNodeById(
      normalizeFileTree(fileTree.value),
      targetFileId,
      (node) => addMaterialToCurrentWeek(node, nextAttachment)
    )

    currentPreviewMaterial.value = nextAttachment
    activeFileId.value = targetFileId
    activeFileName.value = activeFileName.value || findNodeById(fileTree.value, targetFileId)?.name || ''

    const updatedNode = findNodeById(fileTree.value, targetFileId)
    if (isWorkspaceUuid(targetFileId) && Array.isArray(updatedNode?.weeks)) {
      try {
        await saveSessionResources(targetFileId, updatedNode.weeks)
      } catch (error) {
        console.error('[workspace] session resources save failed:', error)
      }
    }
  }

  // 메모 탭의 자료 미리보기 패널을 닫습니다.
  const handleClosePreviewMaterial = () => {
    currentPreviewMaterial.value = null
  }

  // 좌측 폴더에 저장된 첨부 항목을 다시 미리보기로 엽니다.
  const handleOpenStoredMaterial = (materialId) => {
    const target = currentAttachments.value.find((item) => item.id === materialId)
    if (!target) return
    currentPreviewMaterial.value = target
  }

  return {
    currentPreviewMaterial,
    handleUploadLectureMaterials,
    handleClosePreviewMaterial,
    handleOpenStoredMaterial
  }
}
