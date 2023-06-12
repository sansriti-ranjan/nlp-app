import { useState } from "react"

function MyForm(props) {

    return (
      <form>
        <label>
          <input type="text" 
           value={name}
           onChange={(e) => setName(e.target.value)}/>
        </label>
      </form>
    )
  }

export default MyForm