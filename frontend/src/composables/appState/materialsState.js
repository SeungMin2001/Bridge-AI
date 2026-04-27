import { ref } from 'vue'
import { addMaterialToCurrentWeek } from './fileTreeState'

// 워크스페이스에 올린 PDF/PPT 강의자료 첨부와 현재 미리보기 자료를 관리합니다.
export function useMaterialsState({
  fileTree,
  activeFileId,
  activeFileName,
  currentAttachments,
  ensureLectureOneFile,
  updateNodeById
}) {
  const currentPreviewMaterial = ref(null)

  // 선택한 파일을 현재 작업 파일의 이번 주차 강의자료 폴더에 추가하고 바로 미리보기로 엽니다.
  const handleUploadLectureMaterials = (files) => {
    const file = files[0]
    if (!file) return

    const uploadedAt = new Date().toISOString()
    // 현재는 서버/DB 저장이 아니라 브라우저 blob URL 기반 임시 미리보기입니다.
    const nextAttachment = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      name: file.name,
      size: file.size,
      type: file.type,
      uploadedAt,
      url: URL.createObjectURL(file),
      sourceFile: file
    }

    const targetFileId = activeFileId.value || 'lecture-1'

    fileTree.value = updateNodeById(
      ensureLectureOneFile(fileTree.value),
      targetFileId,
      (node) => addMaterialToCurrentWeek(node, nextAttachment)
    )

    currentPreviewMaterial.value = nextAttachment
    activeFileId.value = targetFileId
    activeFileName.value = activeFileName.value || '강의1'
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
