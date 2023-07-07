import { useState, useEffect} from "react"
import axios from 'axios'
import './NLPText.css'

function NLPText() {
  /*Recieve user input and translate on form submit*/

  const [name, setName] = useState("")
  const [sent, setSent] = useState("")

  const handleTextChange = (event) => {
    //console.log(event.target.value)
    setName(event.target.value)
  }

  const handleKeyPress = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
       event.preventDefault();
      console.log('It works')
      handleSubmit(event); // this won't be triggered
    }
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
    <form onSubmit={handleSubmit} className="nlp-text">
      <label>
        <textarea type="text" placeholder="Enter a sentence to translate..." 
          value={name}
          onChange={handleTextChange}
          onKeyDown={handleKeyPress}
          />
          {/*Currently not using the button= with text area, also looks better with button hidden*/}  
        <button type="button">test</button>
        <h1>{sent}</h1>
      </label>
    </form>
  )
}

export default NLPText