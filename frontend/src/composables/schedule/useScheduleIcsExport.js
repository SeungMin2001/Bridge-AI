const escapeIcsText = (value = '') => {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/,/g, '\\,')
    .replace(/;/g, '\\;')
    .replace(/\n/g, '\\n')
}

const parseKoreanTime = (timeText = '오전 09:00') => {
  const match = String(timeText).match(/(오전|오후)\s*(\d{1,2}):(\d{2})/)
  if (!match) return { hour: 9, minute: 0 }

  const [, meridiem, rawHour, rawMinute] = match
  let hour = Number(rawHour)
  const minute = Number(rawMinute)

  if (meridiem === '오전' && hour === 12) hour = 0
  if (meridiem === '오후' && hour !== 12) hour += 12

  return { hour, minute }
}

const formatIcsDateTime = (dateKey, timeText) => {
  const [year, month, day] = String(dateKey).split('-')
  const { hour, minute } = parseKoreanTime(timeText)
  return `${year}${month}${day}T${String(hour).padStart(2, '0')}${String(minute).padStart(2, '0')}00`
}

export function useScheduleIcsExport({ visibleSchedules, formatDateKey, getTypeLabel }) {
  const today = new Date()

  function downloadGoogleCalendarIcs() {
    const items = visibleSchedules.value.filter((item) => item.status === 'confirmed')
    if (!items.length) return

    const nowStamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')
    const events = items.map((item) => {
      const dtStart = formatIcsDateTime(item.dateKey, item.startTime || item.time)
      const dtEnd = formatIcsDateTime(item.dateKey, item.endTime || item.startTime || item.time)
      const description = [
        item.note,
        item.sourceText ? `원문: ${item.sourceText}` : '',
        item.sourceSessionTitle ? `출처: ${item.sourceSessionTitle}` : '',
        `LectoAI 타입: ${getTypeLabel(item.type)}`
      ].filter(Boolean).join('\n')

      return [
        'BEGIN:VEVENT',
        `UID:${item.id}@lectoai.local`,
        `DTSTAMP:${nowStamp}`,
        `DTSTART;TZID=Asia/Seoul:${dtStart}`,
        `DTEND;TZID=Asia/Seoul:${dtEnd}`,
        `SUMMARY:${escapeIcsText(item.title)}`,
        `DESCRIPTION:${escapeIcsText(description)}`,
        'END:VEVENT'
      ].join('\r\n')
    })

    const ics = [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//LectoAI//Schedule Export//KO',
      'CALSCALE:GREGORIAN',
      'METHOD:PUBLISH',
      'X-WR-CALNAME:LectoAI 일정',
      ...events,
      'END:VCALENDAR'
    ].join('\r\n')

    const blob = new Blob([ics], { type: 'text/calendar;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `lectoai-schedules-${formatDateKey(today)}.ics`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  return { downloadGoogleCalendarIcs }
}
