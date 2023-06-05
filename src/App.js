import NLPText from './components/NLPText'
import Navbar from './components/Navbar/Navbar'
import Home from './pages/Home'
import Projects from './pages/Projects'
import Other from './pages/Other'
import About from './pages/About'
import './App.css'
import { Route, Routes } from 'react-router-dom'

function App() {

  return (
    <div>
      <header>
        <Navbar />
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/projects' element={<Projects />} />
          <Route path='/other' element={<Other />} />
          <Route path='/about' element={<About />} />
        </Routes>  
      </header>
    </div>
  );
}

export default App;
