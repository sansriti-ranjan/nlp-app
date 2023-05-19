import { useState } from "react"

function NLPText() {

    const [name, setName] = useState("Enter a sentence...")
    const [title, setTitle] = useState("")

    const handleTextChange = (event) => {
      console.log(event.target.value)
      setName(event.target.value)
    }

    const displayText = (event) => {
      event.preventDefault()
      console.log(event.target)
      setTitle(name)
    }

    return (
      <form onSubmit={displayText}>
        <label>
          <input type="text" 
           value={name}
           onChange={handleTextChange}/>
          <button type="submit">test</button>
          <h1>{title}</h1>
        </label>
      </form>
    )
  }

export default NLPText