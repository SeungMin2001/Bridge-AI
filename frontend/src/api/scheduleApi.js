export const createManualSchedule = async (payload) => {
  const response = await fetch('/schedule/manual', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })

  let result = {}
  try {
    const text = await response.text()
    result = text ? JSON.parse(text) : {}
  } catch (e) {
    result = { error: 'Failed to parse response' }
  }

  if (!response.ok) {
    throw new Error(result.error || result.detail || '일정 수동 추가에 실패했습니다.')
  }

  return result
}
