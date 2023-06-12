import { useState } from 'react'
import { Link } from 'react-router-dom'
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
                  <Link className={item.className} to={item.url}>
                    {item.title}
                  </Link>
                </li>
              )
          })}
        </ul>
      </div>
    </nav>
  )
}

export default Navbar