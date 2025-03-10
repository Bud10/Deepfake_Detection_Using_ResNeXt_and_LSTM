import khecLogo from "../assets/kheclogo.png";
import Image from "next/image";

export default function Header(){
    return(
    <>

  <div className="Header">       
       <div className="font-extrabold text-4xl text-center  m-2 flex items-center justify-center font-serif" style={{color: "#ffffff"}}>
       <Image  src={khecLogo}   width={90}   height={90}  alt="KHEC LOGO"  className="m-2" />
       <span className="self-center">DEEP FAKE DETECTOR</span>
        </div>
        </div>
    </>
    )
}