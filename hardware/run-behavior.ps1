param(
	$MOUSE,
	$EXP,
	$COND,
	$EXPERIMENTER,
	$SESSION=$MOUSE,
	$ID='000',
	$RIG='2',
	$NCAMS=6,
	$GRID=$false,
	$PREVIEW=$false,
	$ANNOTATE=$false
)
if ($MOUSE -eq $null) {
	$MOUSE = Read-Host -Prompt "Forgot to set -MOUSE flag. Please enter a mouse ID (e.g. B6) now"
	if ($SESSION -eq $null) {
		$SESSION=$MOUSE
	}
}
if ($EXP -eq $null) {
	$EXP = Read-Host -Prompt "Forgot to set -EXP flag. Please enter an experiment name now"
}
if ($COND -eq $null) {
	$COND = Read-Host -Prompt "Forgot to set -COND flag. Please enter a condition name now"
}
if ($EXPERIMENTER -eq $null) {
	$EXPERIMENTER = Read-Host -Prompt "Forgot to set -EXPERIMENTER flag. Please enter a name now"
}
$date = Get-Date -format "yyyyMMdd"
$time = Get-Date -format "HH-mm-ss"
$session = "${date}_${SESSION}_${EXP}"
$session_dir = "${session}_rig${RIG}"
$out_dir = "C:\Users\houlab\Documents\Behavior video\$session_dir"
if ($PREVIEW -eq $true) {
	$tl_vid = "NUL.avi"
	$tr_vid = "NUL.avi"
	$l_vid = "NUL.avi"
	$r_vid = "NUL.avi"
	$bc_vid = "NUL.avi"
	$tc_vid = "NUL.avi"
	$concat_vid = "NUL.avi"
	$csv = "NUL.csv"
	$metafile = "NUL.yaml"
} else {
	$tl_vid = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_TL_fullres_${time}.avi"
	$tr_vid = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_TR_fullres_${time}.avi"
	$l_vid = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_L_fullres_${time}.avi"
	$r_vid = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_R_fullres_${time}.avi"
	$bc_vid = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_BC_fullres_${time}.avi"
	$tc_vid = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_TC_fullres_${time}.avi"
	$concat_vid = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_${time}.avi"
	$csv = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_${time}.csv"
	$metafile = "${out_dir}\${date}_${MOUSE}_${EXP}_${COND}_${ID}_${time}_metadata.yaml"
}
$grid_dir = "Z:\Behavior\${session_dir}"
$bitrate = 6000
if ($GRID -eq $true -And $(Test-Path -Path "Z:\Behavior")) {
	New-Item -ItemType Directory -Force -Path $grid_dir
} elseif ($GRID -eq $true) {
	throw "The -GRID flag is true but Z:\Behavior does not exist! Perhaps you forgot to mount the grid."
}
$metadata = @"
experimenter: ${EXPERIMENTER}
num_cameras: ${NCAMS}
"@

$ffmpeg_cmd = "ffmpeg -y " `
+"-f rawvideo -vcodec rawvideo -s 1280x1024 -r 100 -pix_fmt gray8 -i \\.\pipe\bonsai1 " `
+"-f rawvideo -vcodec rawvideo -s 1280x1024 -r 100 -pix_fmt gray8 -i \\.\pipe\bonsai2 " `
+"-f rawvideo -vcodec rawvideo -s 1280x1024 -r 100 -pix_fmt gray8 -i \\.\pipe\bonsai3 " `
+"-f rawvideo -vcodec rawvideo -s 1280x1024 -r 100 -pix_fmt gray8 -i \\.\pipe\bonsai4 " `
+"-f rawvideo -vcodec rawvideo -s 1280x1024 -r 100 -pix_fmt gray8 -i \\.\pipe\bonsai5 " `
+"-f rawvideo -vcodec rawvideo -s 1280x1024 -r 100 -pix_fmt gray8 -i \\.\pipe\bonsai6 " `
+"-filter_complex_script .\ffmpeg_filter.txt " `
+"-map [tl] -b:v ${bitrate}k -pix_fmt gray8 -c:v libx264 -crf 18 -preset ultrafast ""$tl_vid"" " `
+"-map [tr] -b:v ${bitrate}k -pix_fmt gray8 -c:v libx264 -crf 18 -preset ultrafast ""$tr_vid"" " `
+"-map [l] -b:v ${bitrate}k -pix_fmt gray8 -c:v libx264 -crf 18 -preset ultrafast ""$l_vid"" " `
+"-map [r] -b:v ${bitrate}k -pix_fmt gray8 -c:v libx264 -crf 18 -preset ultrafast ""$r_vid"" " `
+"-map [bc] -b:v ${bitrate}k -pix_fmt gray8 -c:v libx264 -crf 18 -preset ultrafast ""$bc_vid"" " `
+"-map [tc] -b:v ${bitrate}k -pix_fmt gray8 -c:v libx264 -crf 18 -preset ultrafast ""$tc_vid"""

if ($PREVIEW -eq $false) {
	New-Item -ItemType Directory -Force -Path $out_dir
	Set-Content `
		-Path $metafile `
		-Value $metadata
}
Start-Process `
	-RedirectStandardOutput $csv `
	-ArgumentList ".\Bonsai-config\20230928_Bonsai_Behavior_6cam_hd.bonsai --no-editor" `
	C:\Users\houlab\AppData\Local\Bonsai\Bonsai.exe
Start-Sleep -Seconds 4
cmd /c $ffmpeg_cmd

if ($PREVIEW -eq $false -And $GRID -eq $true) {
	Write-Output "Copying files to grid..."
	Copy-Item -Path $out_dir\* -Destination $grid_dir -Recurse
}
