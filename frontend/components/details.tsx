import Image from "next/image";
import  robotWalk from "../../../assets/robotwalk.png";
import aiRobot from "../../../assets/robotwalkOthers.png"
import FileUploadWithConstraints from "./upload";

export default function Details(){
    return(
        <>
        
       <div className="flex flex-row ">
          <div className="showImage basis-[30%]">
             <Image
             src={aiRobot}
             width={420}
             height={420}
             alt="KHEC LOGO"
             className="m-2 hidden md:block"
             />

          </div>
        <div className="uploadFile basis-[40%] flex flex-col items-center justify-center"  >
       
       <div className="">
       <FileUploadWithConstraints/>
       </div>
       </div>
       <div className="rWalk basis-[30%]">
        <Image 
      src={robotWalk}
      width={420}
      height={420}
      alt="KHEC LOGO"
      className="m-2 hidden md:block"
       />
       </div>
        </div>
        </>
    )
}