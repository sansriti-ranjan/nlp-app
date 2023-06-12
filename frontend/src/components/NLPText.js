import { useState, useEffect} from "react"
import axios from 'axios'

function NLPText() {
  /*Recieve user input and translate on form submit*/

  const [name, setName] = useState("Enter a sentence...")
  const [sent, setSent] = useState("")

  const handleTextChange = (event) => {
    //console.log(event.target.value)
    setName(event.target.value)
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    
    const inputSentence = { text_message: name }
    console.log(event.target)
    axios.post('http://localhost:8000/translate', inputSentence)
    .then(res => setSent(res.data.target))
    .catch(function (error) {
      console.log(error);
    });
  }

  return (
    <form onSubmit={handleSubmit}>
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