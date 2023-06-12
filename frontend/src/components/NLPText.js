import { useState, useEffect} from "react"
import axios from 'axios'

function NLPText() {

  const [name, setName] = useState("Enter a sentence...")
  const [sent, setSent] = useState("")

  const handleTextChange = (event) => {
    //console.log(event.target.value)
    setName(event.target.value)
  }

  const setTranslatedSent = (event) => {
    setSent(event.target.value)
  }

  //useEffect(() => {
  //  const words = { 'sent': name }
  //  axios.post('http://localhost:8000/', words)
  //    .then(res => setTranslatedSent(res.data))
  //}, []);

  const displayText2 = (event) => {
    event.preventDefault()
    console.log(event.target)
    //axios.post('https://localhost:8000/translate', 
    //  {'text_message': name})
    //.then(res => console.log(res))
    setSent(name)
  }

  const displayText = (event) => {
    event.preventDefault()
    //const words = { text_message: 'no .' }

    console.log(event.target)
    //console.log(words)
    const config = { headers: {'Content-Type': 'application/json'} };
    axios.post('http://localhost:8000/translate', { text_message: "no." }, config)
    .then(res => console.log(res))
    .catch(function (error) {
      console.log(error);
    });
    //setSent(name)
  }

  // Send http request
  const translateSent = (event) => {
    
    
  }

  return (
    <form onSubmit={displayText}>
      <label>
        <input type="text" 
          value={name}
          onChange={handleTextChange}/>
        <button type="submit">test</button>
        <h1>{sent}</h1>
      </label>
    </form>
  )
}

export default NLPText