const parseWorkspaceResponse = async (response, fallbackMessage) => {
  const rawResult = await response.text()
  let result = {}

  try {
    result = rawResult ? JSON.parse(rawResult) : {}
  } catch {
    result = { error: rawResult }
  }

  if (!response.ok || !result.ok) {
    throw new Error(result.error || result.detail || `${fallbackMessage} (${response.status})`)
  }

  return result
}

const requestWorkspaceJson = async (endpoint, options = {}, fallbackMessage = '워크스페이스 요청에 실패했습니다.') => {
  const response = await fetch(`/workspace/${endpoint}`, options)
  return parseWorkspaceResponse(response, fallbackMessage)
}

const jsonRequestOptions = (method, payload) => ({
  method,
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(payload)
})

const sanitizeResourceTree = (weeks = []) => {
  return weeks.map((week) => ({
    ...week,
    materials: Array.isArray(week.materials)
      ? week.materials.map(({ sourceFile, ...material }) => material)
      : [],
    recordings: Array.isArray(week.recordings) ? week.recordings : []
  }))
}

export const isWorkspaceUuid = (value = '') => {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value)
}

export const getWorkspaceTree = async () => {
  const result = await requestWorkspaceJson('tree', {}, '워크스페이스 목록을 불러오지 못했습니다.')
  if (!Array.isArray(result.tree)) {
    throw new Error('워크스페이스 목록을 불러오지 못했습니다.')
  }
  return result.tree
}

export const createWorkspaceFolder = async (payload) => {
  const result = await requestWorkspaceJson(
    'courses',
    jsonRequestOptions('POST', payload),
    '워크스페이스 폴더 저장에 실패했습니다.'
  )
  return result.node
}

export const updateWorkspaceFolder = async (folderId, payload) => {
  const result = await requestWorkspaceJson(
    `courses/${folderId}`,
    jsonRequestOptions('PUT', payload),
    '워크스페이스 폴더 수정에 실패했습니다.'
  )
  return result.node
}

export const deleteWorkspaceFolder = async (folderId) => {
  return requestWorkspaceJson(
    `courses/${folderId}`,
    { method: 'DELETE' },
    '워크스페이스 폴더 삭제에 실패했습니다.'
  )
}

export const createWorkspaceFile = async (payload) => {
  const result = await requestWorkspaceJson(
    'sessions',
    jsonRequestOptions('POST', payload),
    '워크스페이스 파일 저장에 실패했습니다.'
  )
  return result.node
}

export const updateWorkspaceFile = async (fileId, payload) => {
  const result = await requestWorkspaceJson(
    `sessions/${fileId}`,
    jsonRequestOptions('PUT', payload),
    '워크스페이스 파일 수정에 실패했습니다.'
  )
  return result.node
}

export const uploadWorkspaceMaterial = async (sessionId, file) => {
  const formData = new FormData()
  formData.append('file', file)

  const result = await requestWorkspaceJson(
    `sessions/${sessionId}/materials`,
    {
      method: 'POST',
      body: formData
    },
    '강의자료 파일 업로드에 실패했습니다.'
  )
  return result.material
}

export const uploadWorkspaceRecording = async (sessionId, file, options = {}) => {
  const formData = new FormData()
  formData.append('file', file)
  if (options.title) formData.append('title', options.title)
  if (Number.isFinite(Number(options.durationSeconds))) {
    formData.append('duration_seconds', String(Number(options.durationSeconds)))
  }

  const result = await requestWorkspaceJson(
    `sessions/${sessionId}/recordings`,
    {
      method: 'POST',
      body: formData
    },
    '음성파일 업로드에 실패했습니다.'
  )
  return result
}

export const transcribeWorkspaceRecording = async (sessionId, recordingId) => {
  return requestWorkspaceJson(
    `sessions/${sessionId}/recordings/${encodeURIComponent(recordingId)}/transcribe`,
    { method: 'POST' },
    '음성파일 전사에 실패했습니다.'
  )
}

export const deleteWorkspaceFile = async (fileId) => {
  return requestWorkspaceJson(
    `sessions/${fileId}`,
    { method: 'DELETE' },
    '워크스페이스 파일 삭제에 실패했습니다.'
  )
}

export const saveSessionResources = async (sessionId, weeks = []) => {
  const result = await requestWorkspaceJson(
    `sessions/${sessionId}/resources`,
    jsonRequestOptions('PUT', {
      weeks: sanitizeResourceTree(weeks)
    }),
    '현재 파일 자료 저장에 실패했습니다.'
  )
  return result.node
}

export const deleteWorkspaceRecordingData = async (sessionId, recordingId) => {
  return requestWorkspaceJson(
    `sessions/${sessionId}/recordings/${encodeURIComponent(recordingId)}`,
    { method: 'DELETE' },
    '녹음본 관련 데이터 삭제에 실패했습니다.'
  )
}
