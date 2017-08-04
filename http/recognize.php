<?php
	$path = tempnam(sys_get_temp_dir(), 'rec_');
	file_put_contents($path, file_get_contents('php://input'));
	echo shell_exec('./score -f ' . $path);
?>
