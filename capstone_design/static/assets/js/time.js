// 시간을 표시하는 함수
// HH:MM:SS
// X요일
// YYYY년 MM월 DD일
function displayClock() {
    var clock = document.getElementById("time");
    var todayDay = document.getElementById("day");
    var todayDate = document.getElementById("date");

    // 요일 배열
    var dayInString = ['일', '월', '화', '수', '목', '금', '토'];

    // 현재 시간을 불러온다`
    var currentDate = new Date();

    // 연도 정보를 저장
    var year = currentDate.getFullYear();
    // 달 정보를 저장 (getMonth() 함수는 0~11을 반환하므로 끝에 1을 더해준다)
    var month = currentDate.getMonth() + 1;
    // 일 정보를 저장
    var date = currentDate.getDate();
    // 요일 정보를 저장 ()
    var day = currentDate.getDay();
    // 시 정보를 저장
    var hour = currentDate.getHours();
    // 분 정보 저장
    var min = currentDate.getMinutes();
    // 초 정보 저장
    var sec = currentDate.getSeconds();

    clock.innerHTML = `${hour<10 ? `0${hour}`:hour}:${min<10 ? `0${min}`:min}:${sec<10 ? `0${sec}`:sec}`;
    todayDay.innerHTML = `${dayInString[day]}요일`;
    todayDate.innerHTML = `${year}년 ${month}월 ${date}일`;
}

// setInterval() 함수로 1초마다 시간을 갱신
function init() {
    setInterval(displayClock, 1000);
}

// 시간 표시 시작
init();