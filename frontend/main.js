const btn = document.getElementById("start"); //버튼 만들어주기.

btn.onclick = async () => { //버튼 눌렀을시
  //alert("THIS IS THE REAL MAIN JS");
  const ws = new WebSocket("ws://100.104.164.84:8000/ws");

  ws.onmessage=async=(event)=>{
    const data=JSON.parse(event.data)
    console.log("[before]",data.raw_text)
    console.log("[after]",data.text)
  }

  ws.onopen=async()=>{
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); //스트림 만들어주고
    const audioContext=new AudioContext()
    //console.log(audioContext.sampleRate);
    await audioContext.audioWorklet.addModule("pcm-worklet.js")

    const source=audioContext.createMediaStreamSource(stream) //스트림을 소스로 바꿔주고

    const processor=new AudioWorkletNode(audioContext,"pcm-worklet")
    //analyser=audioContext.createAnalyser() //분석 만들어주고
    //source.connect(analyser)

    source.connect(processor)
    processor.connect(audioContext.destination)
    
    processor.port.onmessage = (event) => {
      const data=float32ToInt16(event.data)  //float32->int16
      ws.send(data.buffer)
    };
  }

};

function float32ToInt16(float32Array) {
  const int16Array = new Int16Array(float32Array.length);

  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 32768 : s * 32767;
  }

  return int16Array;
}
