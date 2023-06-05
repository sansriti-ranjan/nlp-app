import { useState } from "react"
import logo from '../../images/default.svg';
import { MenuItems } from './MenuItems'
import './Navbar.css'

const Navbar = () => {
  const [clicked, setClicked] = useState(false)
  


  return (
    <nav className='NavbarItems'>
      <div className='menu-icon'>
        <img className='navbar-icon'
          alt='logo'
          src={logo}
          width="50"
          height="50">
        </img>
      </div>
      <div>
        <ul className='nav-menu active'>
          {MenuItems.map((item, index) => {
              return (
                <li key={index}>
                  <a className={item.className} href={item.url}>
                    {item.title}
                  </a>
                </li>
              )
          })}
        </ul>
      </div>
    </nav>
  )
}

export default Navbar