import { useState } from "react"

function MyForm() {

    const [name, setName] = useState("Enter a sentence")

    const handleTextChange = (event) => {
      setName(event.target.value)
    }

    return (
      <form>
        <label>
          <input type="text" 
           value={name}
           onChange={handleTextChange}/>
        </label>
      </form>
    )
  }

export default MyForm